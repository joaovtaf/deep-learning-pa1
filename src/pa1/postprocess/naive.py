"""Extracao ingenua de instancias: limiar + componentes conexos (Parte 1, item 2).

E de proposito o decoder mais burro possivel. Ele nao consegue separar objeto
encostado, e mostrar isso quantitativamente e o ponto do item 5.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

STRUCT = np.ones((3, 3), dtype=bool)  # conectividade 8


def drop_small_instances(labels: np.ndarray, min_size: int) -> np.ndarray:
    """Tira instancia com menos de min_size pixels e renumera o resto."""
    labels = np.asarray(labels)
    if min_size <= 0 or labels.max() == 0:
        return labels.astype(np.int32)
    areas = np.bincount(labels.ravel())
    remap = np.zeros(areas.size, dtype=np.int32)
    next_id = 1
    for lab in range(1, areas.size):
        if areas[lab] >= min_size:
            remap[lab] = next_id
            next_id += 1
    return remap[labels]


def labels_from_probability(
    prob: np.ndarray,
    threshold: float = 0.5,
    min_size: int = 10,
    fill_holes: bool = True,
) -> np.ndarray:
    """Mapa de probabilidade de foreground (H, W) vira imagem de labels."""
    binary = np.asarray(prob) > threshold
    if fill_holes:
        binary = ndi.binary_fill_holes(binary)
    labels, _ = ndi.label(binary, structure=STRUCT)
    return drop_small_instances(labels.astype(np.int32), min_size)
