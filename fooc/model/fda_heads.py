from typing_extensions import OrderedDict
import torch
from torch import nn, autograd as nn, autograd
from torch.nn.modules import flatten

from detectron2.utils.registry import Registry

from loss import ImageFDALoss, InstanceFDALoss

IMG_FDA_HEAD_REGISTRY = Registry("IMG_FDA_HEAD")
INSTANCE_FDA_HEAD_REGISTRY = Registry("INSTANCE_FDA_HEAD")


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

        num_domains = cfg.FOOC.NUM_DOMAINS
        GRL_gamma = cfg.MODEL.IMG_FDA_HEAD.GRL_GAMMA
        loss_lambda = cfg.MODEL.IMG_FDA_HEAD.LOSS_LAMBDA
        loss_type = cfg.MODEL.IMG_FDA_HEAD.LOSS_TYPE

        # ----- Module Architecture ----- #
        self.GRLs = nn.ModuleDict()  # Grad Reversal Layers
        self.convs1 = nn.ModuleDict()  # First Convolutional Layers
        self.relus = nn.ModuleDict()  # ReLU Layers
        self.convs2 = nn.ModuleDict()  # Second Convolutional Layers

        for idx, shape in enumerate(input_shapes):
            self.GRLs.append(
                {'IMG_GRL_' + str(idx): GradientReversalLayer(GRL_gamma)},
            )
            self.convs1.append(
                {'IMG_convs1_' + str(idx): nn.Conv2d(
                    in_channels=shape,
                    out_channels=shape,
                    kernel_size=1,
                    stride=1
                )},
            )
            self.relus.append(
                {'IMG_ReLU_' + str(idx): nn.ReLU()},
            )
            self.convs2.append(
                {'IMG_convs2_' + str(idx): nn.Conv2d(
                    in_channels=shape,
                    out_channels=num_domains,
                    kernel_size=1,
                    stride=1
                )},
            )

        # ----- Initialization -----#
        for name, conv in [*self.convs1, *self.convs2]:
            torch.nn.init.normal_(conv.weight, std=0.01)
            torch.nn.init.constant_(conv.bias, 0)

            if cfg.MODEL.IMG_FDA_HEAD.SN:
                conv = nn.utils.spectral_norm(
                    conv, name='weight', n_power_iterations=1
                )

        # ----- Loss -----#
        self.loss = ImageFDALoss(loss_lambda, loss_type)

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

        return logits, {'loss_img_fda': loss}


@INSTANCE_FDA_HEAD_REGISTRY.register()
class InstanceFDAHead(nn.Module):
    def __init__(self, cfg, input_shape) -> None:
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        num_domains = cfg.FOOC.NUM_DOMAINS
        GRL_gamma = cfg.MODEL.IMG_FDA_HEAD.GRL_GAMMA
        loss_lambda = cfg.MODEL.IMG_FDA_HEAD.LOSS_LAMBDA
        loss_type = cfg.MODEL.IMG_FDA_HEAD.LOSS_TYPE

        # ----- Module Architecture ----- #
        self.GRL = nn.Sequential(
            OrderedDict([
                ('INSTANCE_GRL', GradientReversalLayer(GRL_gamma)),
            ])
        )  # Grad Reversal Layer

        self.fc1 = nn.Sequential(
            OrderedDict([
                ('INSTANCE_flatten', nn.flatten()),
                ('INSTANCE_linear1', nn.Linear(input_shape.channels
                 * input_shape.height * input_shape.width, 1024)),
                ('INSTANCE_ReLU1', nn.ReLU()),
                ('INSTANCE_dropout1', nn.Dropout(p=0.5)),
            ])
        )  # First FC Layer

        self.fc2 = nn.Sequential(
            OrderedDict([
                ('INSTANCE_linear2', nn.Linear(1024, 1024)),
                ('INSTANCE_ReLU2', nn.ReLU()),
                ('INSTANCE_dropout2', nn.Dropout(p=0.5)),
            ])
        )  # Second FC Layer

        self.logits = nn.Sequential(
            OrderedDict([
                ('INSTANCE_logits', nn.Linear(1024, num_domains)),
            ])
        )  # Logits Layer

        # ----- Initialization -----#
        for layer in [self.fc1[1], self.fc2[0], self.logits[0]]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

            if cfg.MODEL.INSTANCE_FDA_HEAD.SN:
                layer = nn.utils.spectral_norm(
                    layer, name='weight', n_power_iterations=1
                )

        # ----- Loss -----#
        self.loss = InstanceFDALoss(loss_lambda, loss_type)

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
