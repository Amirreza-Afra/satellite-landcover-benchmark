import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from .registry import register_model


@register_model("resnet18")
class ResNet18Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        num_classes = cfg["model"]["num_classes"]
        self.model = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(
            in_features,
            num_classes
        )
        print("Pretrained ResNet18 Loaded")

    def forward(self, x):
        return self.model(x)