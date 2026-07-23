from .registry import SCHEDULER_REGISTRY


def build_scheduler(cfg, optimizer):

    name = cfg["scheduler"]["name"]
    if name not in SCHEDULER_REGISTRY:
        raise ValueError(
            f"Scheduler '{name}' not found"
        )

    scheduler = SCHEDULER_REGISTRY[name]
    return scheduler.build(
        optimizer,
        cfg
    )