"""Encoder ResNet pre-treinado na ImageNet.

So o encoder vem do torchvision, o que o PA permite. torchvision.models.detection nao e
importado em lugar nenhum. O decoder e nosso, ta em unet.py.

output_stride controla o truque do DeepLab (slides 39 a 42): trocar o stride 2 dos
ultimos estagios por dilatacao 2 mantem a resolucao do feature map profundo sem perder
campo receptivo e sem adicionar um peso sequer. E o que a correcao da Parte 5 usa.
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

# output stride -> dilatacao aplicada em (layer3, layer4). 1 quer dizer sem mexer.
DILATION = {32: (1, 1), 16: (1, 2), 8: (2, 4)}


def _dilate_stage(stage, dilation: int) -> None:
    """Troca o stride 2 do estagio por dilatacao, no lugar.

    O torchvision so aceita replace_stride_with_dilation em Bottleneck, e ResNet18/34
    usam BasicBlock, que levanta NotImplementedError. Entao a gente mexe nas convs na
    mao. Os pesos nao mudam de forma, entao o pre-treino da ImageNet continua valendo:
    e exatamente o argumento do slide 39, filtro dilatado e o mesmo filtro com buraco.
    """
    for i, block in enumerate(stage):
        if i == 0:
            block.conv1.stride = (1, 1)
            if block.downsample is not None:
                block.downsample[0].stride = (1, 1)
        for conv in (block.conv1, block.conv2):
            if conv.kernel_size == (3, 3):
                conv.dilation = (dilation, dilation)
                conv.padding = (dilation, dilation)


class ResNetEncoder(nn.Module):
    """Quebra a ResNet em 5 estagios. Com output_stride 32 os strides sao 2, 4, 8, 16, 32."""

    def __init__(self, name: str = "resnet34", pretrained: bool = True, in_channels: int = 3,
                 output_stride: int = 32):
        super().__init__()
        if name not in RESNETS:
            raise KeyError(f"encoder '{name}' nao existe, tem {sorted(RESNETS)}")
        if output_stride not in DILATION:
            raise KeyError(f"output_stride {output_stride} nao existe, tem {sorted(DILATION)}")
        ctor, weights, channels = RESNETS[name]
        net = ctor(weights=weights if pretrained else None)
        d3, d4 = DILATION[output_stride]
        if d3 > 1:
            _dilate_stage(net.layer3, d3)
        if d4 > 1:
            _dilate_stage(net.layer4, d4)

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
        self.output_stride = output_stride
        # stride de cada um dos 5 feature maps, o decoder usa isso pra saber quando
        # precisa fazer upsample e quando os dois mapas ja tem o mesmo tamanho
        self.strides = (2, 4, 8, min(16, output_stride), output_stride)

    def forward(self, x: Tensor) -> list[Tensor]:
        f0 = self.stem(x)                # stride 2
        f1 = self.layer1(self.pool(f0))  # stride 4
        f2 = self.layer2(f1)             # stride 8
        f3 = self.layer3(f2)             # stride 16, ou 8 se dilatado
        f4 = self.layer4(f3)             # stride 32, ou 16/8 se dilatado
        return [f0, f1, f2, f3, f4]


def build_encoder(name: str = "resnet34", pretrained: bool = True, in_channels: int = 3,
                  output_stride: int = 32) -> ResNetEncoder:
    return ResNetEncoder(name=name, pretrained=pretrained, in_channels=in_channels,
                         output_stride=output_stride)
