import torch.optim.lr_scheduler as lr_scheduler

from .registry import register_scheduler


@register_scheduler("cosine")
class CosineScheduler:

    @staticmethod
    def build(optimizer, cfg):

        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["train"]["epochs"]
        )