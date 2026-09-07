"""Checagens da fusao de instancias entre tiles (Parte 4).

Sao casos montados na mao, sem rede nenhuma: a gente finge o resultado de cada tile
e checa se a costura faz o que devia. Assim o teste separa o bug da fusao do erro do
modelo.
"""

from __future__ import annotations

import numpy as np

from pa1.tiling import fuse_tile_instances, paste_per_tile_labels, tile_positions


def test_tile_positions_cover_everything():
    for length, tile, overlap in [(256, 128, 32), (300, 128, 32), (100, 128, 32), (512, 256, 64)]:
        pos = tile_positions(length, tile, overlap)
        assert pos[0] == 0
        assert pos[-1] + tile >= length, f"nao cobriu o fim em {length}/{tile}/{overlap}"


def test_object_on_the_seam_is_split_by_the_naive_paste():
    """O que o item 3 da Parte 4 pede pra mostrar: a emenda parte o objeto em dois."""
    h = w = 128
    tile, overlap = 96, 32
    coords = [(0, 0), (0, 32)]

    # um objeto que vive de x=50 a x=78, ou seja, em cima da emenda
    tiles = []
    for _, x0 in coords:
        t = np.zeros((tile, tile), dtype=np.int32)
        lo, hi = max(0, 50 - x0), min(tile, 78 - x0)
        if hi > lo:
            t[40:70, lo:hi] = 1
        tiles.append(t)

    pasted = paste_per_tile_labels(tiles, coords, (h, w), tile, overlap)
    assert pasted.max() == 2, "o jeito ingenuo tinha que produzir dois pedacos"

    fused = fuse_tile_instances(tiles, coords, (h, w), tile, iou_threshold=0.25)
    assert fused.max() == 1, "a fusao tinha que devolver o objeto inteiro"


def test_fusion_does_not_merge_two_genuinely_different_objects():
    """Dois objetos vizinhos e encostados nao podem virar um so."""
    h = w = 128
    tile = 96
    coords = [(0, 0), (0, 32)]

    tiles = []
    for _, x0 in coords:
        t = np.zeros((tile, tile), dtype=np.int32)
        for lab, (a, b) in enumerate([(40, 55), (55, 70)], start=1):
            lo, hi = max(0, a - x0), min(tile, b - x0)
            if hi > lo:
                t[40:70, lo:hi] = lab
        tiles.append(t)

    fused = fuse_tile_instances(tiles, coords, (h, w), tile, iou_threshold=0.5)
    assert fused.max() == 2, f"tinha que continuar com 2 objetos, virou {fused.max()}"


def test_fusion_is_transitive_across_three_tiles():
    """Objeto que aparece em tres tiles nao pode sair como dois."""
    h, w = 64, 200
    tile = 96
    coords = [(0, 0), (0, 52), (0, 104)]

    tiles = []
    for _, x0 in coords:
        t = np.zeros((tile, tile), dtype=np.int32)
        lo, hi = max(0, 30 - x0), min(tile, 170 - x0)
        if hi > lo:
            t[20:40, lo:hi] = 1
        tiles.append(t)

    fused = fuse_tile_instances(tiles, coords, (h, w), tile, iou_threshold=0.2)
    assert fused.max() == 1
