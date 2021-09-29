from typing_extensions import OrderedDict
import torch
from torch import nn, autograd as nn, autograd
from torch.nn.modules import flatten

from detectron2.utils.registry import Registry

from loss import ImageFDALoss, InstanceFDALoss

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
    def __init__(self, cfg, input_shape) -> None:
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        num_domains = cfg.FOOC.NUM_DOMAINS
        GRL_gamma = cfg.MODEL.IMG_FDA_HEAD.GRL_GAMMA
        loss_lambda = cfg.MODEL.IMG_FDA_HEAD.LOSS_LAMBDA
        loss_type = cfg.MODEL.IMG_FDA_HEAD.LOSS_TYPE
        focal_gamma = cfg.MODEL.IMG_FDA_HEAD.FOCAL_GAMMA

        # ----- Module Architecture ----- #
        self.GRL = nn.Sequential(OrderedDict([
            ('IMG_GRL', GradientReversalLayer(GRL_gamma)),
        ]))  # Grad Reversal Layer

        self.conv1 = nn.Sequential(OrderedDict([
            ('IMG_conv1', nn.Conv2d(in_channels=input_shape.channels,
                                    out_channels=512,
                                    kernel_size=1,
                                    stride=1)),
        ]))  # First Convolutional Layers

        self.relu = nn.Sequential(OrderedDict([
            ('IMG_ReLU', nn.ReLU()),
        ]))   # ReLU Layers

        self.conv2 = nn.Sequential(OrderedDict([
            ('IMG_conv2', nn.Conv2d(in_channels=512,
                                    out_channels=num_domains if num_domains != 2 else 1,
                                    kernel_size=1,
                                    stride=1)),
        ]))  # Second Convolutional Layers

        # ----- Initialization -----#
        for layer in [self.conv1[0], self.conv2[0]]:
            nn.init.normal_(layer.weight, std=0.001)
            nn.init.constant_(layer.bias, 0)

            if cfg.MODEL.INSTANCE_FDA_HEAD.SN:
                layer = nn.utils.spectral_norm(layer, name='weight', n_power_iterations=1)

        # ----- Loss -----#
        self.loss = ImageFDALoss(loss_lambda, loss_type, focal_gamma)

        self.to(self.device)

    def forward(self, features, targets, skip_GRL: bool = False):
        if not self.training:
            return [], {}

        x = features
        if not skip_GRL:
            x = self.GRL(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        logits = x

        loss = self.loss(logits, targets)

        return logits, {'loss_img_fda': loss}


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
            ('INSTANCE_flatten', nn.flatten()),
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

        return logits, {"loss_instance_fda": loss}
