"""Decoder U-Net (slides 25 a 29), escrito por nos.

A head e configuravel de proposito: na Parte 1 e 1 canal (foreground), e na Parte 2
vira 3 classes de fronteira + 1 canal de distancia sem precisar mexer no encoder nem
no decoder, que e exatamente o que o enunciado pede ("mantenham o encoder-decoder da
Parte 1 e mudem o que ele preve"). A flag skip=False deixa o decoder no estilo FCN,
sem skip connection, e context= plugga o modulo do eixo 3 no topo do encoder.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .context import build_context
from .encoders import build_encoder


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    """Upsample 2x, concatena o skip, duas convs.

    upsample=False e o caso do encoder dilatado: com output stride 16 o layer4 sai no
    mesmo tamanho do layer3, entao subir 2x aqui so pra descer de novo seria desperdicio.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, upsample: bool = True):
        super().__init__()
        self.upsample = upsample
        self.conv1 = ConvBNReLU(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x: Tensor, skip: Tensor | None = None) -> Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:  # imagem com lado impar
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class UNet(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        in_channels: int = 3,
        out_channels: int = 1,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
        skip: bool = True,
        pretrained: bool = True,
        context: str = "none",
        context_kwargs: dict | None = None,
        output_stride: int = 32,
    ):
        super().__init__()
        self.encoder = build_encoder(encoder_name, pretrained=pretrained, in_channels=in_channels,
                                     output_stride=output_stride)
        self.skip = skip
        enc_ch = list(self.encoder.out_channels)

        self.context = build_context(context, enc_ch[4], **(context_kwargs or {}))
        self.context_name = (context or "none").lower()

        # do mais profundo pro mais raso, o ultimo estagio nao tem skip
        skip_ch = [enc_ch[3], enc_ch[2], enc_ch[1], enc_ch[0], 0] if skip else [0] * 5

        # o bloco k sobe do estagio (4-k) pro (3-k). so precisa de upsample se os dois
        # tiverem stride diferente, que deixa de valer quando o encoder e dilatado
        st = list(self.encoder.strides) + [1]
        ups = [st[4 - k] != st[3 - k] for k in range(4)] + [True]

        blocks = []
        in_ch = self.context.out_channels
        for out_ch, s_ch, up in zip(decoder_channels, skip_ch, ups):
            blocks.append(DecoderBlock(in_ch, s_ch, out_ch, upsample=up))
            in_ch = out_ch
        self.decoder = nn.ModuleList(blocks)
        self.head = nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1)
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        size = x.shape[-2:]
        f0, f1, f2, f3, f4 = self.encoder(x)
        skips = [f3, f2, f1, f0, None] if self.skip else [None] * 5

        y = self.context(f4)
        for block, s in zip(self.decoder, skips):
            y = block(y, s)

        y = self.head(y)
        if y.shape[-2:] != size:
            y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
        return y


def split_boundary_output(y: Tensor):
    """Separa a saida da cabeca da trilha A em (logits de 3 classes, mapa de distancia).

    Os 3 primeiros canais sao fundo/interior/fronteira e o quarto e a distancia,
    passada por sigmoid porque o alvo e normalizado em [0,1] por instancia.
    """
    return y[:, :3], torch.sigmoid(y[:, 3:4])


def build_model(cfg: dict) -> nn.Module:
    cfg = dict(cfg)
    name = cfg.pop("name", "unet").lower()
    if name != "unet":
        raise KeyError(f"modelo '{name}' nao existe ainda, so tem 'unet'")
    if "decoder_channels" in cfg:
        cfg["decoder_channels"] = tuple(cfg["decoder_channels"])
    return UNet(**cfg)
