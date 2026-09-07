"""Decodificacao da trilha A: watershed com os interiores previstos como marcadores.

Por que isso resolve o problema da Parte 1. Componentes conexos so sabe olhar a
mascara binaria, e dois nucleos encostados formam um blob unico e conexo, entao nao
existe informacao na mascara que permita separa-los. A trilha A muda o que a rede
preve: ela passa a marcar explicitamente a casca de cada nucleo como uma terceira
classe. O interior, que e o que sobra depois de tirar a casca, ja vem desconectado
entre dois nucleos vizinhos. Com isso componentes conexos sobre o interior ja da o
numero certo de objetos, e o watershed so precisa crescer cada marcador de volta ate
a borda real, usando o mapa de distancia como relevo.

O relevo e -dist porque o watershed do skimage enche bacias a partir dos minimos: o
centro do nucleo, onde a distancia e maxima, tem que virar o fundo da bacia.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from .naive import STRUCT, drop_small_instances


def labels_from_boundary_head(
    class_probs,
    dist=None,
    interior_threshold: float = 0.5,
    fg_threshold: float = 0.5,
    min_size: int = 20,
    min_marker_size: int = 4,
    fill_holes: bool = True,
) -> np.ndarray:
    """(3, H, W) de probabilidade por classe + (H, W) de distancia -> mapa de labels.

    class_probs[0] fundo, [1] interior, [2] fronteira.

    interior_threshold controla quantos marcadores nascem, e e o hiperparametro que
    mais mexe no resultado: alto demais mata nucleo pequeno (vira FN), baixo demais
    deixa dois marcadores dentro do mesmo nucleo e ele racha em dois (vira FP).

    fg_threshold usa interior + fronteira, nao so interior, porque a mascara do
    watershed tem que cobrir o objeto inteiro, casca inclusive, senao os nucleos
    saem todos erodidos e o IoU nunca passa de 0.5.
    """
    class_probs = np.asarray(class_probs)
    interior_p, boundary_p = class_probs[1], class_probs[2]

    foreground = (interior_p + boundary_p) > fg_threshold
    if fill_holes:
        foreground = ndi.binary_fill_holes(foreground)

    markers_bin = interior_p > interior_threshold
    markers_bin &= foreground
    markers, _ = ndi.label(markers_bin, structure=STRUCT)
    markers = drop_small_instances(markers, min_marker_size)

    if markers.max() == 0:
        # sem marcador nenhum o watershed devolveria imagem vazia, entao cai no
        # comportamento da Parte 1 e pelo menos entrega os blobs
        labels, _ = ndi.label(foreground, structure=STRUCT)
        return drop_small_instances(labels.astype(np.int32), min_size)

    if dist is None:
        relief = -ndi.distance_transform_edt(foreground)
    else:
        relief = -np.asarray(dist, dtype=np.float32)

    labels = watershed(relief, markers=markers, mask=foreground)
    return drop_small_instances(labels.astype(np.int32), min_size)


def labels_from_distance_only(dist, fg_threshold: float = 0.5, peak_threshold: float = 0.6, min_size: int = 20):
    """Variante que ignora a cabeca de classes e decodifica so do mapa de distancia.

    Serve de ablacao barata na apresentacao: mostra quanto da separacao vem da
    classe fronteira e quanto vem so do relevo continuo.
    """
    dist = np.asarray(dist, dtype=np.float32)
    foreground = dist > fg_threshold * dist.max() if dist.max() > 0 else dist > 0
    foreground = ndi.binary_fill_holes(foreground)
    markers, _ = ndi.label(dist > peak_threshold, structure=STRUCT)
    if markers.max() == 0:
        labels, _ = ndi.label(foreground, structure=STRUCT)
        return drop_small_instances(labels.astype(np.int32), min_size)
    labels = watershed(-dist, markers=markers, mask=foreground)
    return drop_small_instances(labels.astype(np.int32), min_size)
