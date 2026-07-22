from .registry import OPTIMIZER_REGISTRY


def build_optimizer(cfg, model):
    name = cfg["optimizer"]["name"]
    optimizer = OPTIMIZER_REGISTRY[name]
    return optimizer.build(model, cfg)