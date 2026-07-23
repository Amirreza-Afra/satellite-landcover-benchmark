import torch.nn as nn

from .registry import register_loss


@register_loss("cross_entropy")
class CrossEntropyLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        return self.loss(outputs, targets)