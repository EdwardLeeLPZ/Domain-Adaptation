import torch
import torch.nn.functional as F

from fvcore.nn import smooth_l1_loss

from detectron2.layers import ShapeSpec
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs, FastRCNNOutputLayers
from detectron2.utils.events import get_event_storage

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
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
        self.smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        self._init_instance_fda_head(cfg, input_shape)

    def _init_instance_fda_head(self, cfg, input_shape):
        self.instance_fda_on = cfg.MODEL.INSTANCE_FDA_ON
        if not self.instance_fda_on:
            return

        pooler_resolution = cfg.MODEL.INSTANCE_FDA_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio = cfg.MODEL.INSTANCE_FDA_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.INSTANCE_FDA_HEAD.POOLER_TYPE

        # If InstanceFDAHead is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.fda_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.fda_head = build_instance_fda_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch. Like the super method, but also does inference
        when needed for visualization or when running other heads in cascaded mode.
        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
            context_vector (Tensor):
        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        pred_class_logits, pred_proposal_deltas = predictions
        del box_features

        outputs = SelectiveFastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.training,
            self.cfg,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

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
            if self.cfg.MODEL.INSTANCE_FDA_HEAD.FILTER_FOREGROUND:
                proposals, _ = select_foreground_proposals(instances, self.num_classes)
            else:
                proposals = instances
            # use either the region proposals or the refined box predictions
            # to ROIAlign, depending on inference mode
            if self.cfg.MODEL.INSTANCE_FDA_HEAD.USE_REFINED:
                boxes = [x.pred_boxes for x in proposals]
            else:
                boxes = [x.proposal_boxes for x in proposals]

            features = [features[f] for f in self.in_features]

            fda_features = self.fda_pooler(features, boxes)  # [M, C, output_size, output_size]

            _, fda_loss = self.fda_head(fda_features, proposals)
            cons_reg_fda_logits, _ = self.fda_head(fda_features, proposals, skip_GRL=True)
            if cons_reg_fda_logits is not None:
                # fda pooler pooler concatenates the features of all batch elements for the postprocessing,
                # so we split them again
                assert sum([len(proposal) for proposal in proposals]
                           ) == cons_reg_fda_logits.size()[0]
                cons_reg_fda_logits = torch.split(
                    cons_reg_fda_logits,
                    split_size_or_sections=[len(proposal) for proposal in proposals],
                    dim=0
                )
            return cons_reg_fda_logits, fda_loss
        else:
            raise NotImplementedError(
                "Instance-Level FDA Head currently only works during training.")

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


class SelectiveFastRCNNOutputs(FastRCNNOutputs):
    """
    Modifies the original FastRCNNOutputs to add the ability to only compute the losses for those domains
    where this is explicitly specified.
    """

    def __init__(
        self, box2box_transform, pred_class_logits, pred_proposal_deltas, proposals, smooth_l1_beta, training, cfg
    ):
        """
        Args:
            See :class:'FastRCNNOutputs'. To determine wether to compute box losses for a given domain
            it additionally takes the config.
        """
        super().__init__(
            box2box_transform, pred_class_logits, pred_proposal_deltas, proposals, smooth_l1_beta,
        )

        self.training = training
        if self.training:  # Only do this during training
            compute_loss = []
            for prop in proposals:
                if not hasattr(prop, "gt_domains"):
                    compute_loss.extend([False] * len(prop))
                else:
                    if prop.gt_domains.unique() == 0:
                        compute_det_loss = cfg.FOOC.SOURCE.COMPUTE_DET_LOSS
                    else:
                        compute_det_loss = cfg.FOOC.TARGET.COMPUTE_DET_LOSS
                    compute_loss.extend([compute_det_loss] * len(prop))

            self.pred_class_logits_filtered = self.pred_class_logits[compute_loss]
            self.pred_proposal_deltas_filtered = self.pred_proposal_deltas[compute_loss]
            self.proposals_filtered = self.proposals[compute_loss]
            self.gt_classes_filtered = self.gt_classes[compute_loss]
            self.gt_boxes_filtered = self.gt_boxes[compute_loss]
            self.compute_loss = compute_loss

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.
        Returns:
            scalar Tensor
        """
        self._log_accuracy()
        # print(self.pred_class_logits_filtered)
        if len(self.pred_class_logits_filtered) > 0:
            return F.cross_entropy(self.pred_class_logits_filtered, self.gt_classes_filtered, reduction="mean")
        else:
            return self.pred_class_logits.sum() * 0

    def box_reg_loss(self):
        """
        Compute the smooth L1 loss for box regression.
        Returns:
            scalar Tensor
        """
        if not any(self.compute_loss):  # If there are no proposals we want to compute a loss for we skip
            return self.pred_proposal_deltas.sum() * 0

        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor[self.compute_loss], self.gt_boxes.tensor[self.compute_loss]
        )
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1
        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = torch.nonzero((self.gt_classes[self.compute_loss] >= 0) & (
            self.gt_classes[self.compute_loss] < bg_class_ind)).squeeze(1)

        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[self.compute_loss][fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        loss_box_reg = smooth_l1_loss(
            self.pred_proposal_deltas[self.compute_loss][fg_inds[:, None], gt_class_cols],
            gt_proposal_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )
        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes[self.compute_loss].numel()
        return loss_box_reg

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        if not any(self.compute_loss):  # Nothing to do here if there is no gt
            return

        num_instances = self.gt_classes[self.compute_loss].numel()
        pred_classes = self.pred_class_logits[self.compute_loss].argmax(dim=1)
        bg_class_ind = self.pred_class_logits[self.compute_loss].shape[1] - 1

        fg_inds = (self.gt_classes[self.compute_loss] >= 0) & (
            self.gt_classes[self.compute_loss] < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[self.compute_loss][fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes[self.compute_loss]).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
        if num_fg > 0:
            storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg)
            storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_fg)
