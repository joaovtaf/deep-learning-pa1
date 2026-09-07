"""Modulos de contexto global (eixo 3 da Parte 3).

Sao os dois mecanismos dos slides 51 a 55, plugados no topo do encoder, antes do
decoder. A pergunta do enunciado e se contexto global ajuda a separar instancias ou
so a classifica-las melhor, e pra responder isso os dois tem que entrar no mesmo
lugar da rede, mudando so o modulo.

ParseNet (slides 51 e 52): media global do feature map, projetada e somada de volta
em todo pixel. O pixel passa a saber a estatistica da imagem inteira.

PSPNet (slides 53 a 55): em vez de um unico numero por canal, faz pooling em varias
grades (1x1, 2x2, 3x3, 6x6), projeta cada uma e concatena tudo de volta. Da contexto
em varias escalas em vez de so global.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Identity(nn.Module):
    """Sem contexto, que e o braco de controle do eixo 3."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x: Tensor) -> Tensor:
        return x


class ImagePooling(nn.Module):
    """ParseNet. Media global, 1x1 conv, broadcast e concatena."""

    def __init__(self, in_channels: int, channels: int = 128):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = in_channels

    def forward(self, x: Tensor) -> Tensor:
        pooled = F.adaptive_avg_pool2d(x, 1)
        # BatchNorm num mapa 1x1 explode com batch 1, entao no eval nao tem problema
        # mas no treino o batch precisa ser > 1, que e o caso
        g = self.project(pooled)
        g = g.expand(-1, -1, x.shape[-2], x.shape[-1])
        return self.fuse(torch.cat([x, g], dim=1))


class PyramidPooling(nn.Module):
    """PSPNet. Pooling em varias grades, projeta cada nivel e concatena."""

    def __init__(self, in_channels: int, bins=(1, 2, 3, 6), channels: int | None = None):
        super().__init__()
        channels = channels or max(in_channels // len(bins), 32)
        self.bins = tuple(bins)
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, channels, 1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
                for _ in self.bins
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + channels * len(self.bins), in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = in_channels

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        parts = [x]
        for b, stage in zip(self.bins, self.stages):
            p = F.adaptive_avg_pool2d(x, b)
            p = stage(p)
            parts.append(F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False))
        return self.fuse(torch.cat(parts, dim=1))


CONTEXTS = {"none": Identity, "image_pooling": ImagePooling, "pyramid_pooling": PyramidPooling}


def build_context(name: str, in_channels: int, **kwargs) -> nn.Module:
    name = (name or "none").lower()
    if name not in CONTEXTS:
        raise KeyError(f"contexto '{name}' nao existe, tem {sorted(CONTEXTS)}")
    if name == "none":
        return Identity(in_channels)
    return CONTEXTS[name](in_channels, **kwargs)
