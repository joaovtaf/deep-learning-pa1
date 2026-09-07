"""Dataset sintetico da Parte 0.

Imagens 128x128 com 5 a 20 elipses, tamanho variado, ruido e contraste variaveis,
e muitas delas encostando. O ponto de encostar e o que faz o teste unitario ter
valor: se as elipses ficassem soltas, componentes conexos ja resolveria e a Parte 1
nao teria fracasso nenhum pra mostrar.

Nao guarda nada em disco, cada indice gera a partir de uma seed derivada, entao o
dataset e deterministico e os splits ficam disjuntos so mudando a seed base.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage as ndi
from torch.utils.data import Dataset

from .targets import distance_map, semantic_mask, three_class_mask


def _draw_ellipse(labels, cy, cx, ry, rx, angle, label):
    """Pinta uma elipse no mapa de labels, sobrescrevendo quem estava embaixo."""
    h, w = labels.shape
    rad = max(ry, rx) + 2
    y0, y1 = max(0, int(cy - rad)), min(h, int(cy + rad) + 1)
    x0, x1 = max(0, int(cx - rad)), min(w, int(cx + rad) + 1)
    if y0 >= y1 or x0 >= x1:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1]
    dy, dx = yy - cy, xx - cx
    ca, sa = np.cos(angle), np.sin(angle)
    u = (dx * ca + dy * sa) / rx
    v = (-dx * sa + dy * ca) / ry
    labels[y0:y1, x0:x1][u * u + v * v <= 1.0] = label


def _place_centers(rng, size, n_target, radius_range, touch_prob=0.7):
    """Sorteia centro e raio de cada elipse, ancorando a maioria numa ja colocada."""
    placed = []
    for _ in range(n_target):
        ry = rng.uniform(*radius_range)
        rx = ry * rng.uniform(0.55, 1.7)
        angle = rng.uniform(0, np.pi)

        if placed and rng.random() < touch_prob:
            # cola na anterior: distancia entre centros um pouco menor que a soma
            # dos raios medios, entao as duas se tocam ou se sobrepoem de leve
            ay, ax, ary, arx, _ = placed[rng.integers(len(placed))]
            theta = rng.uniform(0, 2 * np.pi)
            reach = (0.5 * (ary + arx) + 0.5 * (ry + rx)) * rng.uniform(0.85, 1.08)
            cy = ay + reach * np.sin(theta)
            cx = ax + reach * np.cos(theta)
            cy = float(np.clip(cy, ry * 0.4, size - ry * 0.4))
            cx = float(np.clip(cx, rx * 0.4, size - rx * 0.4))
        else:
            cy = rng.uniform(ry * 0.5, size - ry * 0.5)
            cx = rng.uniform(rx * 0.5, size - rx * 0.5)

        placed.append((cy, cx, ry, rx, angle))
    return placed


def _relabel_sequential(labels, min_area=0):
    """Renumera pra 1..n, jogando fora quem ficou menor que min_area.

    O filtro de area importa: elipse desenhada por cima enterra a anterior e o que
    sobra dela as vezes e uma lasca de 6 px. Lasca nao e objeto, e so atrapalha a
    metrica (nem o oraculo consegue casar com ela), entao some.
    """
    areas = np.bincount(labels.ravel())
    present = np.nonzero(areas >= max(min_area, 1))[0]
    present = present[present > 0]
    lut = np.zeros(max(int(labels.max()), 0) + 1, dtype=np.int32)
    lut[present] = np.arange(1, present.size + 1, dtype=np.int32)
    return lut[labels], int(present.size)


def generate_sample(rng, size=128, n_range=(5, 20), radius_range=(5, 18), min_area=30):
    """Devolve (imagem float32 em [0,1], labels int32 renumerado 1..n)."""
    lo, hi = n_range
    n_target = int(rng.integers(lo, hi + 1))

    # elipse desenhada por cima pode enterrar uma anterior, entao conta quantas
    # sobrevivem e desenha mais ate cair na faixa pedida
    labels = np.zeros((size, size), dtype=np.int32)
    n_alive = 0
    for attempt in range(12):
        need = n_target - n_alive
        if need <= 0:
            break
        for cy, cx, ry, rx, angle in _place_centers(rng, size, need, radius_range):
            _draw_ellipse(labels, cy, cx, ry, rx, angle, labels.max() + 1)
        labels, n_alive = _relabel_sequential(labels, min_area)

    # se passou do teto (nao costuma passar), apaga as ultimas
    if n_alive > hi:
        labels[labels > hi] = 0
        labels, n_alive = _relabel_sequential(labels, min_area)

    # intensidade por instancia, contraste e fundo variaveis
    fg_level = rng.uniform(0.45, 0.95)
    bg_level = rng.uniform(0.02, 0.30)
    image = np.full((size, size), bg_level, dtype=np.float32)
    for lab in range(1, n_alive + 1):
        m = labels == lab
        image[m] = fg_level * rng.uniform(0.75, 1.15)

    image = ndi.gaussian_filter(image, sigma=rng.uniform(0.6, 1.8))
    image += rng.normal(0.0, rng.uniform(0.01, 0.09), size=image.shape)
    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    return image, labels


class SyntheticEllipsesDataset(Dataset):
    def __init__(
        self, length=512, size=128, seed=0, n_range=(5, 20), radius_range=(5, 18), boundary_thickness=2, min_area=30
    ):
        self.length = int(length)
        self.size = int(size)
        self.seed = int(seed)
        self.n_range = tuple(n_range)
        self.radius_range = tuple(radius_range)
        self.boundary_thickness = int(boundary_thickness)
        self.min_area = int(min_area)
        self.image_ids = [f"synth_{seed}_{i:05d}" for i in range(self.length)]

    def __len__(self):
        return self.length

    def raw(self, idx):
        # seed por indice, entao dois datasets com a mesma seed base dao a mesma imagem
        rng = np.random.default_rng((self.seed + 1) * 1_000_003 + idx)
        return generate_sample(rng, self.size, self.n_range, self.radius_range, self.min_area)

    def __getitem__(self, idx):
        image, labels = self.raw(idx)
        return {
            "image": torch.from_numpy(np.repeat(image[None], 3, axis=0)),
            "mask": torch.from_numpy(semantic_mask(labels)[None]),
            "labels": torch.from_numpy(labels.astype(np.int64)),
            "three": torch.from_numpy(three_class_mask(labels, self.boundary_thickness).astype(np.int64)),
            "dist": torch.from_numpy(distance_map(labels)[None]),
            "image_id": self.image_ids[idx],
            "orig_size": torch.tensor([labels.shape[0], labels.shape[1]]),
        }
