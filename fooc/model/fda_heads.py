from typing_extensions import OrderedDict
import torch
from torch import nn as nn
from torch import autograd as autograd
from torch.nn.modules import flatten

from detectron2.utils.registry import Registry

from .loss import ImageFDALoss, InstanceFDALoss

__all__ = [
    "build_instance_fda_head",
    "build_img_fda_head",
    "GradientReversalLayer",
    "ImageFDAHead",
    "InstanceFDAHead",
]

IMG_FDA_HEAD_REGISTRY = Registry("IMG_FDA_HEAD")
INSTANCE_FDA_HEAD_REGISTRY = Registry("INSTANCE_FDA_HEAD")


def build_instance_fda_head(cfg, input_shape):
    """
    Build a ROI-level feature distribution alignment head
    defined by `cfg.MODEL.ROI_FDA_HEAD.NAME`.
    """
    head_name = cfg.MODEL.INSTANCE_FDA_HEAD.NAME
    return INSTANCE_FDA_HEAD_REGISTRY.get(head_name)(cfg, input_shape)


def build_img_fda_head(cfg, input_shape):
    """
    Build a image-level feature distribution alignment head
    defined by `cfg.MODEL.IMG_FDA_HEAD.NAME`.
    """
    head_name = cfg.MODEL.IMG_FDA_HEAD.NAME
    return IMG_FDA_HEAD_REGISTRY.get(head_name)(cfg, input_shape)


class GradientReversalLayer(nn.Module):
    """
    Flips and scales the gradients during backpropagation
    as described in https://arxiv.org/pdf/1505.07818.pdf.
    """

    def __init__(self, gamma=1):
        super().__init__()
        self.register_buffer('gamma', torch.Tensor([gamma]))

    def forward(self, x):
        return _GradientReverser.apply(x, self.gamma)


class _GradientReverser(autograd.Function):

    @staticmethod
    def forward(ctx, x, gamma):
        ctx.save_for_backward(gamma)
        return x.clone()

    @staticmethod
    def backward(ctx, gradients):
        gamma, = ctx.saved_tensors
        return gradients * -gamma, None


@IMG_FDA_HEAD_REGISTRY.register()
class ImageFDAHead(nn.Module):
    def __init__(self, cfg, input_shapes) -> None:
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.in_features = cfg.MODEL.IMG_FDA_HEAD.IN_FEATURES

        num_domains = cfg.FOOC.NUM_DOMAINS
        GRL_gamma = cfg.MODEL.IMG_FDA_HEAD.GRL_GAMMA
        loss_lambda = cfg.MODEL.IMG_FDA_HEAD.LOSS_LAMBDA
        loss_type = cfg.MODEL.IMG_FDA_HEAD.LOSS_TYPE
        focal_gamma = cfg.MODEL.IMG_FDA_HEAD.FOCAL_GAMMA

        in_channels = [input_shapes[f].channels for f in self.in_features]

        # ----- Module Architecture ----- #
        self.GRLs = nn.ModuleList()  # Grad Reversal Layers
        self.convs1 = nn.ModuleList()  # First Convolutional Layers
        self.relus = nn.ModuleList()  # ReLU Layers
        self.convs2 = nn.ModuleList()  # Second Convolutional Layers

        for in_channel in in_channels:
            self.GRLs.append(GradientReversalLayer(GRL_gamma),)
            self.convs1.append(nn.Conv2d(
                in_channels=in_channel, out_channels=512, kernel_size=1, stride=1),
            )
            self.relus.append(nn.ReLU(),)
            self.convs2.append(nn.Conv2d(
                in_channels=512, out_channels=num_domains if num_domains != 2 else 1, kernel_size=1, stride=1),
            )

        # ----- Initialization -----#
        for layer in [*self.convs1, *self.convs2]:
            torch.nn.init.normal_(layer.weight, std=0.001)
            torch.nn.init.constant_(layer.bias, 0)

            if cfg.MODEL.IMG_FDA_HEAD.SN:
                layer = nn.utils.spectral_norm(
                    layer, name='weight', n_power_iterations=1
                )

        # ----- Loss -----#
        self.loss = ImageFDALoss(loss_lambda, loss_type, focal_gamma)

        self.to(self.device)

    def forward(self, features, targets, skip_GRL: bool = False):
        if not self.training:
            return [], {}

        logits = []
        for idx, f in enumerate(self.in_features):
            x = features[f]
            if not skip_GRL:
                x = self.GRLs[idx](x)
            x = self.convs1[idx](x)
            x = self.relus[idx](x)
            x = self.convs2[idx](x)
            logits.append(x)

        loss = self.loss(logits, targets)

        return logits, {'loss_fda_image': loss}


@INSTANCE_FDA_HEAD_REGISTRY.register()
class InstanceFDAHead(nn.Module):
    def __init__(self, cfg, input_shape) -> None:
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        num_domains = cfg.FOOC.NUM_DOMAINS
        GRL_gamma = cfg.MODEL.INSTANCE_FDA_HEAD.GRL_GAMMA
        loss_lambda = cfg.MODEL.INSTANCE_FDA_HEAD.LOSS_LAMBDA
        loss_type = cfg.MODEL.INSTANCE_FDA_HEAD.LOSS_TYPE
        focal_gamma = cfg.MODEL.INSTANCE_FDA_HEAD.FOCAL_GAMMA

        # ----- Module Architecture ----- #
        self.GRL = nn.Sequential(OrderedDict([
            ('INSTANCE_GRL', GradientReversalLayer(GRL_gamma)),
        ]))  # Grad Reversal Layer

        self.fc1 = nn.Sequential(OrderedDict([
            ('INSTANCE_flatten', nn.Flatten(start_dim=1, end_dim=-1)),
            ('INSTANCE_linear1', nn.Linear(input_shape.channels
                                           * input_shape.height * input_shape.width, 1024)),
            ('INSTANCE_ReLU1', nn.ReLU()),
            ('INSTANCE_dropout1', nn.Dropout(p=0.5)),
        ]))  # First FC Layer

        self.fc2 = nn.Sequential(OrderedDict([
            ('INSTANCE_linear2', nn.Linear(1024, 1024)),
            ('INSTANCE_ReLU2', nn.ReLU()),
            ('INSTANCE_dropout2', nn.Dropout(p=0.5)),
        ]))  # Second FC Layer

        self.logits = nn.Sequential(OrderedDict([
            ('INSTANCE_logits', nn.Linear(1024, num_domains if num_domains != 2 else 1)),
        ]))  # Logits Layer

        # ----- Initialization -----#
        for idx, layer in enumerate([self.fc1[1], self.fc2[0], self.logits[0]]):
            nn.init.normal_(layer.weight, std=0.01 if idx < 2 else 0.05)
            nn.init.constant_(layer.bias, 0)

            if cfg.MODEL.INSTANCE_FDA_HEAD.SN:
                layer = nn.utils.spectral_norm(layer, name='weight', n_power_iterations=1)

        # ----- Loss -----#
        self.loss = InstanceFDALoss(loss_lambda, loss_type, focal_gamma)

        self.to(self.device)

    def forward(self, features, targets, skip_GRL: bool = False):
        if not self.training:
            return 0, {}

        x = features
        if not skip_GRL:
            x = self.GRL(x)
        x = self.fc1(x)
        x = self.fc2(x)
        logits = self.logits(x)

        loss = self.loss(logits, targets)

        return logits, {"loss_fda_instance": loss}


class DAFRCNNConsistReg:
    """
    Computes the consistency regularization loss given by DAFRCNN https://arxiv.org/pdf/1803.03243.pdf,  formula (8)
    """

    def __init__(self, cfg):
        self.loss_weight = cfg.MODEL.FDA_CONSISTENCY_REGULARIZATION.LOSS_WEIGHT
        self.loss_norm = cfg.MODEL.FDA_CONSISTENCY_REGULARIZATION.LOSS_NORM
        self.size_average = cfg.MODEL.FDA_CONSISTENCY_REGULARIZATION.SIZE_AVERAGE

    def __call__(self, img_fda_logits, instance_fda_logits):
        losses = fda_consistency_regularization(
            img_fda_logits, instance_fda_logits, loss_norm=self.loss_norm, size_average=self.size_average
        )

        losses = {
            name: loss * self.loss_weight
            for name, loss in losses.items()
        }
        return losses


def fda_consistency_regularization(img_fda_logits, instance_fda_logits, loss_norm, size_average):
    """
    Computes the consistency regularization loss as stated in the paper
    `Domain Adaptive Faster R-CNN for Object Detection in the Wild`
    L_cst = \sum_{i,j}||\frac{1}{|I|}\sum_{u,v}p_i^{(u,v)}-p_{i,j}||_2
    https://arxiv.org/pdf/1803.03243.pdf,  formula (8)
    Pytorch implementation: https://github.com/krumo/Domain-Adaptive-Faster-RCNN-PyTorch
    Args:
        img_fda_logits(list): list of logits of the image level domain discriminators  [N, 1, H, W],
                              one of each feature map
        instance_fda_logits: list of the logits of the instance level domain discriminator  [NUM_BOXES, 1],
                        one for each batch element
        loss_norm(str): the distance criterion for instance and image-level probs, one of ["l1", "l2"]
        size_average(bool): flag that indicates whether loss portions are averaged or summed.
    Returns:
        the computed consistency regularization loss (dict[name, Tensor])
    """
    assert loss_norm in ["l1", "l2"], loss_norm
    loss = 0.

    # flatten img fda logits
    img_fda_logits = [torch.flatten(x, start_dim=1)
                      for x in img_fda_logits]  # list of [N, 1 x H x W]
    # activate img fda logits
    img_fda_preds = [torch.sigmoid(x) for x in img_fda_logits]  # list of [N, 1 x H x W]
    # mean per feature map
    img_fda_preds = [torch.mean(x, 1, keepdim=True) for x in img_fda_preds]  # list of [N, 1]
    # average over different feature maps
    img_fda_preds = torch.cat(img_fda_preds, dim=1)  # [N, NUM_FEATURE_MAPS]
    # NOTE: the original implementation computes the consistency regularization only on one image level feature map.
    #       here we support multiple ones and just average them. This means using one image level feature map results
    #       in the same method.
    img_fda_preds = torch.mean(img_fda_preds, dim=1, keepdim=False)  # [N]

    assert len(img_fda_preds) == len(
        instance_fda_logits), f"{len(img_fda_preds)} != {len(instance_fda_logits)}"
    for img_fda_preds_per_image, instance_fda_logits_per_image in zip(img_fda_preds, instance_fda_logits):
        # we must handle the case where no gt are available for an image
        if instance_fda_logits_per_image.size()[0] == 0:
            loss = loss + img_fda_preds_per_image.sum() * 0.
            continue

        # activate roi fda logits
        instance_fda_logits_per_image = torch.sigmoid(instance_fda_logits_per_image)  # [NUM_BOXES, 1]

        # compute consistency loss for current image
        diffs_per_image = img_fda_preds_per_image - instance_fda_logits_per_image  # [NUM_BOXES, 1]
        if loss_norm == "l1":
            losses_per_image = torch.abs(diffs_per_image)  # [NUM_BOXES, 1]
        else:
            losses_per_image = diffs_per_image * diffs_per_image * 0.5  # [NUM_BOXES, 1]

        if size_average:
            # NOTE: we use the mean over the number of box proposals instead of simple summing them up as in the
            #       original implementation to be consistent with the box regression loss. In addition, it stabilizes
            #       the training in the beginning by avoiding high loss in case of a lot of false positives.
            loss_per_image = torch.mean(losses_per_image)
        else:
            loss_per_image = torch.sum(losses_per_image)

        # update loss
        loss = loss + loss_per_image

    return {"loss_fda_consistency_regularization": loss}
