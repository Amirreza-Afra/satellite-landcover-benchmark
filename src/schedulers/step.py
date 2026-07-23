import torch.optim.lr_scheduler as lr_scheduler

from .registry import register_scheduler


@register_scheduler("step")
class StepScheduler:

    @staticmethod
    def build(optimizer, cfg):

        return lr_scheduler.StepLR(
            optimizer,
            step_size=5,
            gamma=0.1
        )