import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageFDALoss(nn.Module):
    def __init__(self, loss_lambda, loss_type):
        super().__init__()
        self.loss_lambda = loss_lambda
        self.loss_type = loss_type

    def forward(self, img_fda_logits, targets):


class InstanceFDALoss(nn.Module):
    def __init__(self, loss_lambda, loss_type):
        super().__init__()
        self.loss_lambda = loss_lambda
        self.loss_type = loss_type

    def forward(self, instance_fda_logits, targets):
        