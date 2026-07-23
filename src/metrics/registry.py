METRIC_REGISTRY = {}


def register_metric(name):

    def decorator(cls):
        METRIC_REGISTRY[name] = cls
        return cls

    return decorator