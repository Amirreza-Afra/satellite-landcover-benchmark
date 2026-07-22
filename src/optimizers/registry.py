OPTIMIZER_REGISTRY = {}


def register_optimizer(name):
    def decorator(cls):
        OPTIMIZER_REGISTRY[name] = cls
        return cls
    return decorator