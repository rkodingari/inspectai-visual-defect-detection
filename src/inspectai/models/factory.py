from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
    resnet18,
)

from inspectai.models.baseline import BaselineCNN

SUPERVISED_MODELS = ("baseline_cnn", "mobilenet_v3_small", "efficientnet_b0", "resnet18")
MODEL_DISPLAY_NAMES = {
    "baseline_cnn": "Baseline CNN",
    "mobilenet_v3_small": "MobileNetV3-Small",
    "efficientnet_b0": "EfficientNet-B0",
    "resnet18": "ResNet18",
}


class MobileNetDefectClassifier(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.network = mobilenet_v3_small(weights=weights)
        if freeze_backbone:
            for parameter in self.network.features.parameters():
                parameter.requires_grad = False
            # Fine-tune the final feature block while freezing most of the backbone.
            for parameter in self.network.features[-1].parameters():
                parameter.requires_grad = True
        in_features = self.network.classifier[-1].in_features
        self.network.classifier[-1] = nn.Linear(in_features, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.network.features[-1]


class EfficientNetDefectClassifier(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.network = efficientnet_b0(weights=weights)
        if freeze_backbone:
            for parameter in self.network.features.parameters():
                parameter.requires_grad = False
            for parameter in self.network.features[-1].parameters():
                parameter.requires_grad = True
        in_features = self.network.classifier[-1].in_features
        self.network.classifier[-1] = nn.Linear(in_features, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.network.features[-1]


class ResNet18DefectClassifier(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.network = resnet18(weights=weights)
        if freeze_backbone:
            for parameter in self.network.parameters():
                parameter.requires_grad = False
            for parameter in self.network.layer4.parameters():
                parameter.requires_grad = True
        self.network.fc = nn.Linear(self.network.fc.in_features, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.network.layer4[-1]


def create_model(name: str, pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    if name == "baseline_cnn":
        return BaselineCNN()
    if name == "mobilenet_v3_small":
        return MobileNetDefectClassifier(pretrained=pretrained, freeze_backbone=freeze_backbone)
    if name == "efficientnet_b0":
        return EfficientNetDefectClassifier(pretrained=pretrained, freeze_backbone=freeze_backbone)
    if name == "resnet18":
        return ResNet18DefectClassifier(pretrained=pretrained, freeze_backbone=freeze_backbone)
    raise ValueError(f"Unknown model '{name}'")


def trainable_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
