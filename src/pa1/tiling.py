"""Parte 4: inferencia em mosaico.

O slide 83 descreve a pratica padrao pra imagem grande: processa em tiles com
patches sobrepostos, considera a parte interna e faz a media dos resultados. Isso
resolve segmentacao semantica porque o que se faz media e um mapa denso de
probabilidade, e probabilidade de duas rodadas do mesmo pixel e comparavel.

Pra instancia isso quebra, e o motivo e simples: o id da instancia e arbitrario. O
nucleo 7 do tile da esquerda e o nucleo 3 do tile da direita podem ser o mesmo
nucleo, e nao existe media entre 7 e 3. Pior, um nucleo cortado pela emenda vira
dois objetos, cada um com metade da area, e nenhum dos dois casa com o ground truth
no limiar de IoU 0.5.

Aqui tem tres estrategias, na ordem em que a apresentacao vai mostrar:

per_tile    decodifica dentro de cada tile e cola, que e o jeito ingenuo e o que
            quebra. E o que a gente mostra falhando.
fuse        mesma coisa, mas depois costura: duas instancias de tiles vizinhos que
            se sobrepoem na faixa comum viram uma so, por union-find.
blend       faz a media dos logits na sobreposicao (o slide 83 ao pe da letra) e so
            depois decodifica, uma vez so, na imagem inteira. So da pra fazer isso
            porque a nossa representacao e densa, um detector com proposta de regiao
            nao teria esse caminho.
"""

from __future__ import annotations

import numpy as np
import torch

from .metrics.instance import instance_iou_matrix
from .utils import as_rgb_uint8


def tile_positions(length: int, tile: int, overlap: int) -> list[int]:
    """Inicios dos tiles cobrindo [0, length) com sobreposicao."""
    if length <= tile:
        return [0]
    step = tile - overlap
    pos = list(range(0, length - tile + 1, step))
    if pos[-1] + tile < length:
        pos.append(length - tile)
    return pos


def _inner_window(start: int, tile: int, length: int, margin: int):
    """Parte interna do tile, que e o pedaco confiavel do slide 83."""
    lo = start + (margin if start > 0 else 0)
    hi = min(start + tile - (margin if start + tile < length else 0), length)
    return lo, hi


@torch.no_grad()
def predict_tiles(model, image, device, tile=256, overlap=64, batch_size=4):
    """Roda o modelo em tiles e devolve (logits por tile, posicoes)."""
    h, w = image.shape[:2]
    ys, xs = tile_positions(h, tile, overlap), tile_positions(w, tile, overlap)
    coords = [(y, x) for y in ys for x in xs]

    crops = []
    for y, x in coords:
        c = image[y : y + tile, x : x + tile]
        ph, pw = tile - c.shape[0], tile - c.shape[1]
        if ph or pw:
            c = np.pad(c, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        crops.append(c)

    outs = []
    model.eval()
    for i in range(0, len(crops), batch_size):
        batch = np.stack(crops[i : i + batch_size]).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        outs.append(model(torch.from_numpy(batch).to(device)).cpu())
    return torch.cat(outs), coords


def blend_logits(logits, coords, shape, tile, overlap, margin=None):
    """Media dos logits na sobreposicao, considerando so a parte interna (slide 83)."""
    h, w = shape
    margin = overlap // 2 if margin is None else margin
    acc = torch.zeros((logits.shape[1], h, w), dtype=torch.float32)
    cnt = torch.zeros((1, h, w), dtype=torch.float32)

    for k, (y, x) in enumerate(coords):
        y0, y1 = _inner_window(y, tile, h, margin)
        x0, x1 = _inner_window(x, tile, w, margin)
        acc[:, y0:y1, x0:x1] += logits[k, :, y0 - y : y1 - y, x0 - x : x1 - x]
        cnt[:, y0:y1, x0:x1] += 1
    return acc / cnt.clamp_min(1)


def paste_per_tile_labels(per_tile_labels, coords, shape, tile, overlap, margin=None):
    """Cola o resultado ja decodificado tile a tile, sem costura nenhuma.

    Cada tile contribui so com sua parte interna, exatamente como o slide manda pra
    segmentacao semantica. O objeto que cai na emenda sai partido em dois.
    """
    h, w = shape
    margin = overlap // 2 if margin is None else margin
    out = np.zeros((h, w), dtype=np.int32)
    next_id = 1
    for k, (y, x) in enumerate(coords):
        y0, y1 = _inner_window(y, tile, h, margin)
        x0, x1 = _inner_window(x, tile, w, margin)
        sub = per_tile_labels[k][y0 - y : y1 - y, x0 - x : x1 - x]
        present = np.unique(sub)
        present = present[present > 0]
        for lab in present:
            out[y0:y1, x0:x1][sub == lab] = next_id
            next_id += 1
    return out


def _relabel(labels):
    present = np.unique(labels)
    present = present[present > 0]
    if present.size == 0:
        return np.zeros_like(labels, dtype=np.int32)
    lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    lut[present] = np.arange(1, present.size + 1, dtype=np.int32)
    return lut[labels]


def fuse_tile_instances(per_tile_labels, coords, shape, tile, iou_threshold=0.25):
    """A correcao: cola os tiles inteiros e costura o que se sobrepoe.

    Cada tile entra inteiro (nao so a parte interna), com ids proprios, e duas
    instancias de tiles diferentes que dividem area viram a mesma. O criterio e IoU
    na faixa de sobreposicao, e a uniao e transitiva via union-find, senao um nucleo
    que aparece em tres tiles viraria dois objetos em vez de um.

    Usar IoU e nao so "encostou" importa: dois nucleos vizinhos de verdade encostam
    e nao podem ser fundidos. O que os separa e que instancias correspondentes tem
    IoU alto na faixa comum, e nucleos vizinhos tem IoU perto de zero.
    """
    h, w = shape
    offset = 0
    stacked = np.zeros((len(coords), h, w), dtype=np.int32)
    for k, (y, x) in enumerate(coords):
        lab = per_tile_labels[k]
        sub_h, sub_w = min(tile, h - y), min(tile, w - x)
        block = lab[:sub_h, :sub_w]
        stacked[k, y : y + sub_h, x : x + sub_w] = np.where(block > 0, block + offset, 0)
        offset = int(max(offset, stacked[k].max()))

    parent: dict[int, int] = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            yi, xi = coords[i]
            yj, xj = coords[j]
            y0, y1 = max(yi, yj), min(min(yi + tile, yj + tile), h)
            x0, x1 = max(xi, xj), min(min(xi + tile, xj + tile), w)
            if y0 >= y1 or x0 >= x1:
                continue
            a = stacked[i, y0:y1, x0:x1]
            b = stacked[j, y0:y1, x0:x1]
            if a.max() == 0 or b.max() == 0:
                continue
            ids_a = np.unique(a)
            ids_a = ids_a[ids_a > 0]
            ids_b = np.unique(b)
            ids_b = ids_b[ids_b > 0]
            iou = instance_iou_matrix(a, b)
            for pi, ia in enumerate(ids_a):
                for pj, ib in enumerate(ids_b):
                    if iou[pi, pj] >= iou_threshold:
                        union(int(ia), int(ib))

    # o tile que chegou primeiro ganha o pixel, entao a costura fica estavel
    merged = np.zeros((h, w), dtype=np.int32)
    for k in range(len(coords)):
        lab = stacked[k]
        m = (lab > 0) & (merged == 0)
        vals = lab[m]
        if vals.size:
            merged[m] = np.array([find(int(v)) for v in vals], dtype=np.int32)
    return _relabel(merged)


def build_mosaic(dataset, indices, grid, tile_size=256):
    """Monta uma imagem grande colando varias imagens do dataset numa grade.

    Cada imagem entra com seus labels deslocados, entao o ground truth do mosaico e
    a uniao dos ground truths, sem id repetido entre celulas da grade.
    """
    rows, cols = grid
    big = np.zeros((rows * tile_size, cols * tile_size, 3), dtype=np.uint8)
    big_labels = np.zeros((rows * tile_size, cols * tile_size), dtype=np.int32)
    offset = 0
    for k, idx in enumerate(indices[: rows * cols]):
        image, labels = dataset.raw(idx)
        image = as_rgb_uint8(image)[:tile_size, :tile_size]
        labels = _relabel(labels[:tile_size, :tile_size])
        r, c = divmod(k, cols)
        y, x = r * tile_size, c * tile_size
        big[y : y + image.shape[0], x : x + image.shape[1]] = image
        big_labels[y : y + labels.shape[0], x : x + labels.shape[1]] = np.where(labels > 0, labels + offset, 0)
        offset += int(labels.max())
    return big, big_labels
