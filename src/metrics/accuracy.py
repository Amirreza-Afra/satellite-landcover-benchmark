import torch

from .registry import register_metric


@register_metric("accuracy")
class Accuracy:

    def __call__(self, outputs, targets):
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == targets).sum().item()
        total = targets.size(0)
        return correct / total