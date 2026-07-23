import torch.optim as optim

from .registry import register_optimizer


@register_optimizer("adamw")
class AdamWOptimizer:

   @staticmethod
   def build(parameters, cfg):
        return optim.AdamW(
            parameters,
            lr=cfg["train"]["learning_rate"],
            weight_decay=cfg["optimizer"]["weight_decay"]
        )