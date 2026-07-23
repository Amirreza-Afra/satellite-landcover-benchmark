SCHEDULER_REGISTRY = {}


def register_scheduler(name):

    def decorator(cls):
        SCHEDULER_REGISTRY[name] = cls
        return cls

    return decorator