"""Campo receptivo teorico do encoder (Parte 5, obrigatorio, slides 35 a 38).

A conta e a recorrencia padrao. Percorrendo as camadas em ordem, com j o espacamento
acumulado (jump) e r o campo receptivo:

    j_out = j_in * s
    r_out = r_in + (k - 1) * d * j_in

k tamanho do kernel, s stride, d a taxa de dilatacao (atrous, slide 39). Comeca com
j = 1 e r = 1 no pixel de entrada.

O slide 38 e exatamente isso: pooling aumenta o campo receptivo sem custar parametro,
mas cobra em resolucao. O slide 39 mostra a alternativa, atrous, que aumenta r sem
mexer em j nem no numero de pesos. As duas contas saem da mesma funcao aqui, e a
comparacao com e sem atrous na mesma resolucao de saida e o que a Parte 5 pede.

Importante pra apresentacao: isso e o campo receptivo teorico. O efetivo e bem menor,
porque o peso da contribuicao cai indo pro centro (Luo et al. 2016, "Understanding the
Effective Receptive Field"). Entao quando um objeto e do tamanho do campo teorico ele
ja esta em apuros muito antes de ultrapassa-lo.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Layer:
    name: str
    kernel: int
    stride: int = 1
    dilation: int = 1


def receptive_field(layers) -> list[dict]:
    """Tabela camada a camada com jump, campo receptivo e stride acumulado."""
    j, r = 1, 1
    rows = []
    for layer in layers:
        r = r + (layer.kernel - 1) * layer.dilation * j
        j = j * layer.stride
        rows.append({"layer": layer.name, "kernel": layer.kernel, "stride": layer.stride,
                     "dilation": layer.dilation, "jump": j, "rf": r})
    return rows


def _basic_block(prefix, stride, dilation=1):
    """Bloco basico da ResNet18/34: duas convs 3x3, a primeira com o stride."""
    return [
        Layer(f"{prefix}.conv1", 3, stride, dilation),
        Layer(f"{prefix}.conv2", 3, 1, dilation),
    ]


def resnet18_layers(dilate_layer4: bool = False, dilate_layer3: bool = False) -> list[Layer]:
    """ResNet18 quebrada em convs que contam pro campo receptivo.

    A conv 1x1 do downsample nao entra porque nao aumenta r. Com dilate_layer4 a
    gente troca o stride 2 do layer4 por dilatacao 2, que e o truque do DeepLab
    (slide 40): mantem o output stride em 16 em vez de 32 e ainda assim continua
    ganhando campo receptivo.
    """
    layers = [Layer("conv1", 7, 2), Layer("maxpool", 3, 2)]
    layers += _basic_block("layer1.0", 1) + _basic_block("layer1.1", 1)
    layers += _basic_block("layer2.0", 2) + _basic_block("layer2.1", 1)

    s3, d3 = (1, 2) if dilate_layer3 else (2, 1)
    layers += _basic_block("layer3.0", s3, d3) + _basic_block("layer3.1", 1, d3)

    s4, d4 = (1, d3 * 2) if dilate_layer4 else (2, d3)
    layers += _basic_block("layer4.0", s4, d4) + _basic_block("layer4.1", 1, d4)
    return layers


def resnet34_layers(dilate_layer4: bool = False, dilate_layer3: bool = False) -> list[Layer]:
    layers = [Layer("conv1", 7, 2), Layer("maxpool", 3, 2)]
    for i in range(3):
        layers += _basic_block(f"layer1.{i}", 1)
    for i in range(4):
        layers += _basic_block(f"layer2.{i}", 2 if i == 0 else 1)

    s3, d3 = (1, 2) if dilate_layer3 else (2, 1)
    for i in range(6):
        layers += _basic_block(f"layer3.{i}", s3 if i == 0 else 1, d3)

    s4, d4 = (1, d3 * 2) if dilate_layer4 else (2, d3)
    for i in range(3):
        layers += _basic_block(f"layer4.{i}", s4 if i == 0 else 1, d4)
    return layers


ENCODERS = {
    "resnet18": resnet18_layers,
    "resnet18_atrous": lambda: resnet18_layers(dilate_layer4=True),
    "resnet18_atrous2": lambda: resnet18_layers(dilate_layer4=True, dilate_layer3=True),
    "resnet34": resnet34_layers,
    "resnet34_atrous": lambda: resnet34_layers(dilate_layer4=True),
    "resnet34_atrous2": lambda: resnet34_layers(dilate_layer4=True, dilate_layer3=True),
}


def encoder_receptive_field(name: str = "resnet18") -> dict:
    rows = receptive_field(ENCODERS[name]())
    return {"name": name, "rows": rows, "rf": rows[-1]["rf"], "output_stride": rows[-1]["jump"]}


def format_table(info: dict) -> str:
    lines = [f"campo receptivo teorico, {info['name']}", f"{'camada':<16}{'k':>3}{'s':>3}{'d':>3}{'jump':>7}{'RF':>7}"]
    for r in info["rows"]:
        lines.append(f"{r['layer']:<16}{r['kernel']:>3}{r['stride']:>3}{r['dilation']:>3}{r['jump']:>7}{r['rf']:>7}")
    lines.append(f"final: RF {info['rf']} px, output stride {info['output_stride']}")
    return "\n".join(lines)


def object_size_stats(labels_iterable) -> dict:
    """Distribuicao de diametro equivalente dos objetos, pra comparar com o RF."""
    import numpy as np

    diameters = []
    for labels in labels_iterable:
        labels = np.asarray(labels)
        if labels.max() == 0:
            continue
        areas = np.bincount(labels.ravel())[1:]
        areas = areas[areas > 0]
        diameters += list(2.0 * np.sqrt(areas / np.pi))
    d = np.array(diameters)
    return {
        "n": int(d.size),
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "p95": float(np.percentile(d, 95)),
        "p99": float(np.percentile(d, 99)),
        "max": float(d.max()),
        "diameters": d,
    }
