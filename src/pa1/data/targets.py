"""Alvos derivados das mascaras individuais (Parte 2, trilha A).

O enunciado pergunta tres coisas nessa trilha e as respostas estao todas aqui.

Como gerar o rotulo de fronteira a partir das mascaras individuais?
    Erodindo cada instancia separadamente. O que sobra da erosao e interior
    (classe 1) e a casca que a erosao comeu e fronteira (classe 2). Fazer isso por
    instancia e o que separa dois nucleos encostados: no contato existe uma casca
    de cada lado, entao a classe 2 aparece exatamente onde componentes conexos
    fundiria os dois. Se a gente erodisse a mascara semantica inteira de uma vez,
    o contato ficaria no meio do foreground e nao viraria fronteira nenhuma.

Que espessura?
    thickness em pixels de raio da erosao. Fina demais e a rede nao consegue prever
    (uma casca de 1 px some no downsample de stride 32 do encoder), grossa demais
    come o interior de nucleo pequeno e ele deixa de virar marcador no watershed.
    O DSB2018 tem nucleo de raio 5 px, entao thickness 2 ja consome boa parte deles.
    O default e 2 e o eixo de espessura ta exposto no config.

Como pesar a classe fronteira, que e minoritaria?
    Duas alavancas. class_frequencies + alpha_from_frequencies da o peso alpha da
    CE balanceada (slides 74 e 75), e touching_weight_map da um peso por pixel
    extra em cima do contato entre instancias, que e a ideia do mapa de peso do
    proprio paper da U-Net. O eixo 2 da Parte 3 varre justamente essas alavancas.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

BACKGROUND, INTERIOR, BOUNDARY = 0, 1, 2


def semantic_mask(labels) -> np.ndarray:
    """Mascara binaria fundo vs objeto, que e o alvo do baseline da Parte 1."""
    return (np.asarray(labels) > 0).astype(np.float32)


def _instance_edt(labels):
    """Distancia ao fundo dentro de cada instancia, calculada instancia a instancia.

    Fazer no bounding box de cada uma evita o custo de uma EDT global por objeto e,
    mais importante, impede que a distancia vaze de um nucleo pro vizinho encostado.
    """
    labels = np.asarray(labels)
    dist = np.zeros(labels.shape, dtype=np.float32)
    if labels.max() == 0:
        return dist

    for lab, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        # uma borda de 1 px em volta pra EDT saber onde e o fundo
        pad = tuple(slice(max(0, s.start - 1), min(n, s.stop + 1)) for s, n in zip(sl, labels.shape))
        sub = labels[pad] == lab
        if not sub.any():
            continue
        d = ndi.distance_transform_edt(np.pad(sub, 1, mode="constant"))[1:-1, 1:-1]
        dist[pad] = np.maximum(dist[pad], d.astype(np.float32))
    return dist


def distance_map(labels, normalize: bool = True) -> np.ndarray:
    """Mapa continuo de distancia ao fundo, 0 no fundo e 1 no centro da instancia.

    E o relevo do watershed e o alvo da perda L1/L2 do slide 80. Normalizar por
    instancia e escolha nossa: sem isso o nucleo grande domina a regressao e o
    pequeno vira ruido, e o que a decodificacao precisa e a forma do morro, nao a
    altura absoluta em pixels.
    """
    dist = _instance_edt(labels)
    if not normalize:
        return dist

    labels = np.asarray(labels)
    out = np.zeros_like(dist)
    for lab, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        m = labels[sl] == lab
        peak = dist[sl][m].max() if m.any() else 0.0
        if peak > 0:
            block = out[sl]
            block[m] = dist[sl][m] / peak
    return out


def three_class_mask(labels, thickness: int = 2) -> np.ndarray:
    """0 fundo, 1 interior, 2 fronteira. A erosao e por instancia, nao na mascara toda."""
    labels = np.asarray(labels)
    out = np.zeros(labels.shape, dtype=np.uint8)
    fg = labels > 0
    out[fg] = BOUNDARY  # tudo que e objeto comeca como casca

    if thickness <= 0:
        out[fg] = INTERIOR
        return out

    dist = _instance_edt(labels)
    interior = fg & (dist > thickness)

    # nucleo menor que a espessura sumiria inteiro e deixaria de virar marcador no
    # watershed, entao pra esses a gente guarda o pico da distancia como interior
    for lab, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        m = labels[sl] == lab
        if not m.any() or interior[sl][m].any():
            continue
        sub = np.where(m, dist[sl], 0.0)
        interior[sl] |= m & (sub >= sub.max())

    out[interior] = INTERIOR
    return out


def touching_weight_map(labels, thickness: int = 2, w_touch: float = 5.0, radius: int = 3) -> np.ndarray:
    """Peso por pixel, maior no contato entre duas instancias diferentes.

    Um pixel entra no contato se, dentro de um raio pequeno, aparece mais de um id
    de instancia. E o pedaco de fronteira que realmente importa: a casca externa de
    um nucleo isolado nao separa nada, mas a casca entre dois encostados separa.
    """
    labels = np.asarray(labels)
    weight = np.ones(labels.shape, dtype=np.float32)
    if labels.max() < 2:
        return weight

    size = 2 * radius + 1
    hi = ndi.maximum_filter(labels, size=size)
    lo = ndi.minimum_filter(np.where(labels > 0, labels, np.iinfo(np.int32).max), size=size)
    touching = (hi > 0) & (lo != np.iinfo(np.int32).max) & (hi != lo)
    weight[touching] = w_touch
    return weight


def class_frequencies(labels_iterable, thickness: int = 2) -> np.ndarray:
    """Fracao de pixel de cada uma das 3 classes, pra calibrar o alpha da CE balanceada."""
    counts = np.zeros(3, dtype=np.float64)
    for labels in labels_iterable:
        three = three_class_mask(labels, thickness)
        counts += np.bincount(three.ravel(), minlength=3)
    return counts / max(counts.sum(), 1.0)


def alpha_from_frequencies(freq, mode: str = "inverse", clip: float = 20.0) -> list[float]:
    """Peso por classe do slide 74, maior pra classe menos abundante.

    "inverse" usa 1/f e "sqrt_inverse" usa 1/sqrt(f), que e mais suave. O clip
    existe porque com fronteira em torno de 8% dos pixels o 1/f puro ja passa de 12
    e a loss vira quase so fronteira.
    """
    freq = np.asarray(freq, dtype=np.float64)
    freq = np.maximum(freq, 1e-8)
    w = 1.0 / freq if mode == "inverse" else 1.0 / np.sqrt(freq)
    w = w / w.mean()
    return [float(v) for v in np.clip(w, 1.0 / clip, clip)]
