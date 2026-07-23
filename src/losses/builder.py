from .registry import LOSS_REGISTRY


def build_loss(cfg):
    loss_name = cfg["loss"]["name"]
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(f"Loss '{loss_name}' is not registered.")

    return LOSS_REGISTRY[loss_name]()