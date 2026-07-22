from .registry import register_model


@register_model("resnet18")
class ResNet18Model:

    def __init__(self, cfg):

        self.cfg = cfg

        print("ResNet18 Model Initialized")