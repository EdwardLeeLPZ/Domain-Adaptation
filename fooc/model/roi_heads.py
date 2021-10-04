import torch

from detectron2.layers import ShapeSpec
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler

from .fda_heads import build_instance_fda_head

@ROI_HEADS_REGISTRY.register()
class DomainAdaptiveROIHeads(StandardROIHeads):
    """
    Extends the standard detectron2 ROI-Heads with an instance-level
    feature distrbution alignment head.
    """
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.cfg = cfg
        self._init_instance_fda_head(cfg)

    def _init_instance_fda_head(self, cfg):
        self.instance_fda_on = cfg.MODEL.INSTANCE_FDA_ON
        if not self.instance_fda_on:
            return

        pooler_resolution = cfg.MODEL.INSTANCE_FDA_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio = cfg.MODEL.INSTANCE_FDA_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.INSTANCE_FDA_HEAD.POOLER_TYPE

        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.fda_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.fda_head = build_instance_fda_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )
    
    def _forward_fda(self, features, instances):
        """
        Implements forward logic for the Instance-level feature distribution alignment head.
        Args:
            features (list[Tensor]): FPN feature maps.
            instances (list[Instances]): per image instances (proposals).
        Returns:
            roi_fda_logits (list of Tensor): one logits tensor per image of size [NUM_BOXES, 1]
            roi_fda_loss (dict[Tensor]).
        """
        if not self.instance_fda_on:
            return None, {} if self.training else instances

        if self.training:
            proposals = instances
            # use either the region proposals or the refined box predictions
            # to ROIAlign, depending on inference mode
            if self.cfg.MODEL.INSTANCE_FDA_HEAD.USE_REFINED:
                boxes = [x.pred_boxes for x in proposals]
            else:
                boxes = [x.proposal_boxes for x in proposals]

            fda_features = self.fda_pooler(features, boxes)  # [M, C, output_size, output_size]

            _, fda_loss = self.fda_head(fda_features, proposals)
            cons_reg_fda_logits, _ = self.fda_head(fda_features, proposals, skip_GRL=True)
            if cons_reg_fda_logits is not None:
                # fda pooler pooler concatenates the features of all batch elements for the postprocessing,
                # so we split them again
                assert sum([len(proposal) for proposal in proposals]) == cons_reg_fda_logits.size()[0]
                cons_reg_fda_logits = torch.split(
                    cons_reg_fda_logits,
                    split_size_or_sections=[len(proposal) for proposal in proposals],
                    dim=0
                )
            return cons_reg_fda_logits, fda_loss
        else:
            raise NotImplementedError("Instance-Level FDA Head currently only works during training.")
    
    def forward(self, images, features, proposals, targets=None):
        """
        Simply extends StandardROIHeads.forward by additionally performing the forward pass
        for the fda head, if it exists.
        """
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            cons_reg_fda_logits, fda_instance_losses = self._forward_fda(features, proposals)
            losses.update(fda_instance_losses)
            return proposals, losses, cons_reg_fda_logits
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}