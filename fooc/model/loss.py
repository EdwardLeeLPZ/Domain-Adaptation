import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn.focal_loss import sigmoid_focal_loss

from detectron2.layers.wrappers import cat


def _rezise_targets(targets, height, width):
    """
    Resize the targets to the same size of logits (add height and width channels)
    """
    targets = torch.stack(targets)  # convert list of scalar tensors to single tensor
    targets = targets.view(
        targets.size(0), targets.size(1), 1, 1,  # add two dummy spatial dims
    )
    targets = targets.expand(
        targets.size(0), targets.size(1), height, width   # expand to spatial size of x
    )
    return targets


class ImageFDALoss(nn.Module):
    def __init__(self, loss_lambda, loss_type='cross_entropy', focal_gamma=2):
        super().__init__()
        self.loss_lambda = loss_lambda
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma

    def forward(self, img_fda_logits, targets):
        losses = []
        for logits in img_fda_logits:
            targets_resized = _rezise_targets(targets, logits.size(-2), logits.size(-1))

            if self.loss_type == "cross_entropy":
                loss = F.binary_cross_entropy_with_logits(
                    logits, targets_resized, reduction='mean',
                )
            elif self.loss_type == "l2":
                scores = torch.sigmoid(logits)
                loss = torch.norm(scores - targets_resized, p=2, dim=1).mean()
            elif self.loss_type == "focal":
                loss = sigmoid_focal_loss(
                    logits, targets_resized, gamma=self.focal_gamma, reduction="mean"
                )
            else:
                raise ValueError(f"Unsupported loss type \"{self.loss_type}\"")

            losses.append(loss)

        return sum(losses) / (len(losses) + 1e-8) * self.loss_lambda


class InstanceFDALoss(nn.Module):
    def __init__(self, loss_lambda, loss_type='cross_entropy', focal_gamma=2):
        super().__init__()
        self.loss_lambda = loss_lambda
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma

    def forward(self, instance_fda_logits, instances):
        gt_domains = []
        for instances_per_img in instances:
            if len(instances_per_img) > 0 and hasattr(instances_per_img, "gt_domains"):
                gt_domains_per_img = instances_per_img.gt_domains.unsqueeze(-1)
                gt_domains.append(gt_domains_per_img)
                # Sanity check: All instances in an image should have the same domain label
                assert gt_domains_per_img.unique().numel() == 1

        # if there is no ground truth, there is no loss to compute
        if len(gt_domains) == 0 or instance_fda_logits.shape != cat(gt_domains, dim=0).shape:
            return instance_fda_logits.sum() * 0

        if self.loss_type == "cross_entropy":
            loss = F.binary_cross_entropy_with_logits(
                instance_fda_logits, cat(gt_domains, dim=0), reduction='mean',
            )
        elif self.loss_type == "focal":
            loss = sigmoid_focal_loss(
                instance_fda_logits, cat(gt_domains, dim=0), gamma=self.focal_gamma, reduction="mean"
            )
        else:
            raise ValueError(f"Unsupported loss type \"{self.loss_type}\"")

        return loss * self.loss_lambda
