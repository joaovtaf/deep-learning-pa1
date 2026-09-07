"""Parte 6: teste de estresse por corrupcao.

Escolhemos a opcao das corrupcoes porque ela da uma curva de degradacao de verdade,
com eixo x ordenado, em vez de dois pontos soltos. Sao tres familias, cada uma em
tres intensidades, aplicadas so na hora de avaliar. O modelo nunca ve corrupcao no
treino, tirando o brilho/contraste leve que ja esta no augment, e essa assimetria e
justamente o ponto do teste.

Tudo em numpy sobre a imagem uint8, antes de virar tensor, pra corromper exatamente
o que a rede receberia na vida real.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

SEVERITIES = (1, 2, 3)


def blur(image, severity=1, rng=None):
    sigma = {1: 1.0, 2: 2.0, 3: 4.0}[severity]
    out = np.stack([ndi.gaussian_filter(image[..., c].astype(np.float32), sigma) for c in range(image.shape[2])], -1)
    return np.clip(out, 0, 255).astype(np.uint8)


def noise(image, severity=1, rng=None):
    std = {1: 8.0, 2: 20.0, 3: 40.0}[severity]
    rng = rng or np.random.default_rng(0)
    out = image.astype(np.float32) + rng.normal(0.0, std, size=image.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def brightness_contrast(image, severity=1, rng=None):
    """Escurece e achata o contraste ao mesmo tempo, que e o caso ruim de microscopia."""
    gain, bias = {1: (0.75, -10.0), 2: (0.55, -25.0), 3: (0.35, -40.0)}[severity]
    mean = image.astype(np.float32).mean()
    out = (image.astype(np.float32) - mean) * gain + mean + bias
    return np.clip(out, 0, 255).astype(np.uint8)


CORRUPTIONS = {"blur": blur, "noise": noise, "brightness_contrast": brightness_contrast}


def apply_corruption(image, name, severity, rng=None):
    if name == "none":
        return image
    return CORRUPTIONS[name](image, severity, rng)


def rescale(image, labels, factor):
    """Reamostra imagem e labels. Serve pro teste de escala, se a gente quiser mostrar.

    Labels vao com ordem 0 (vizinho mais proximo), senao a interpolacao inventaria
    ids que nao existem entre duas instancias vizinhas.
    """
    zoom_img = (factor, factor, 1)
    img = ndi.zoom(image.astype(np.float32), zoom_img, order=1)
    lab = ndi.zoom(labels, (factor, factor), order=0)
    return np.clip(img, 0, 255).astype(np.uint8), lab.astype(np.int32)
