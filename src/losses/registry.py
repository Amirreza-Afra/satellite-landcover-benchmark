LOSS_REGISTRY = {}


def register_loss(name):

    def decorator(loss_class):
        LOSS_REGISTRY[name] = loss_class
        return loss_class

    return decorator