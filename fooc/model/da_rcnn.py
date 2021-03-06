import logging

from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

from .fda_heads import build_img_fda_head, DAFRCNNConsistReg

@META_ARCH_REGISTRY.register()
class DomainAdaptiveRCNN(GeneralizedRCNN):
    """
    Main class for domain-adaptive RCNN-based architectures. Currently supports
    Image and Instance-level feature discrimination.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self._init_img_fda_head(cfg)
        self._init_fda_consistency_regularization(cfg)
    
    def _init_img_fda_head(self, cfg):
        self.img_fda_on = cfg.MODEL.IMG_FDA_ON
        if not self.img_fda_on:
            return

        self.img_fda_head = build_img_fda_head(
            cfg, self.backbone.output_shape()
        )
    
    def _init_fda_consistency_regularization(self, cfg):
        self.fda_consistency_regularization_on = cfg.MODEL.FDA_CONSISTENCY_REGULARIZATION_ON
        if self.fda_consistency_regularization_on:
            assert cfg.MODEL.IMG_FDA_ON, "Image FDA must be available for consistency regularization but isn't."
            assert cfg.MODEL.INSTANCE_FDA_ON, "ROI FDA must be available for consistency regularization but isn't."
            self.fda_consistency_regularization = DAFRCNNConsistReg(cfg)
    
    def forward(self, batched_inputs):
        """
        Extends super().forward() with the domain-adversarial forward and some additional
        functionality. Ideally this would be a wrapper around the super method but we need to
        access intermediate results.
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)
        losses = {}

        # preprocessing and backbone
        images = self.preprocess_image(batched_inputs)
        if 'domain' in batched_inputs[0]:
            gt_domains = [x['domain'].to(self.device) for x in batched_inputs]
        else:
            gt_domains = None
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        
        features = self.backbone(images.tensor)

        # proposals
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
        losses.update(proposal_losses)

        if self.img_fda_on:
            _, fda_image_losses = self.img_fda_head(features, gt_domains)
            losses.update(fda_image_losses)

            if self.fda_consistency_regularization_on:
                cons_reg_img_fda_logits, _ = self.img_fda_head(features, gt_domains, skip_GRL=True)

        # roi heads
        _, detector_losses, cons_reg_instance_fda_logits = self.roi_heads(images, features, proposals, gt_instances)
        losses.update(detector_losses)

        if self.fda_consistency_regularization_on:
            consistency_regularization_loss = self.fda_consistency_regularization(
                cons_reg_img_fda_logits, cons_reg_instance_fda_logits)
            losses.update(consistency_regularization_loss)
        return losses