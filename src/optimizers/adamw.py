import torch.optim as optim

from .registry import register_optimizer


@register_optimizer("adamw")
class AdamWOptimizer:

    @staticmethod
    def build(model, cfg):

        return optim.AdamW(
            model.parameters(),
            lr=cfg["train"]["learning_rate"],
            weight_decay=cfg["optimizer"]["weight_decay"]

        )