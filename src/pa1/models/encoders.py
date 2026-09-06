"""Encoder ResNet pre-treinado na ImageNet.

So o encoder vem do torchvision, o que o PA permite. torchvision.models.detection
nao e importado em lugar nenhum. O decoder e nosso, ta em unet.py.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision import models

# nome: (construtor, pesos, canais dos 5 feature maps devolvidos)
RESNETS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, (64, 64, 128, 256, 512)),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, (64, 64, 128, 256, 512)),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, (64, 256, 512, 1024, 2048)),
}


class ResNetEncoder(nn.Module):
    """Quebra a ResNet em 5 estagios, com strides 2, 4, 8, 16 e 32."""

    def __init__(self, name: str = "resnet34", pretrained: bool = True, in_channels: int = 3):
        super().__init__()
        if name not in RESNETS:
            raise KeyError(f"encoder '{name}' nao existe, tem {sorted(RESNETS)}")
        ctor, weights, channels = RESNETS[name]
        net = ctor(weights=weights if pretrained else None)

        if in_channels != 3:
            # media dos pesos RGB replicada, pra nao jogar fora o pre-treino
            old = net.conv1
            net.conv1 = nn.Conv2d(in_channels, old.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                with torch.no_grad():
                    net.conv1.weight[:] = old.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.pool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.out_channels = channels

    def forward(self, x: Tensor) -> list[Tensor]:
        f0 = self.stem(x)                # stride 2
        f1 = self.layer1(self.pool(f0))  # stride 4
        f2 = self.layer2(f1)             # stride 8
        f3 = self.layer3(f2)             # stride 16
        f4 = self.layer4(f3)             # stride 32
        return [f0, f1, f2, f3, f4]


def build_encoder(name: str = "resnet34", pretrained: bool = True, in_channels: int = 3) -> ResNetEncoder:
    return ResNetEncoder(name=name, pretrained=pretrained, in_channels=in_channels)
