"""Checagens da trilha A: alvo de fronteira, mapa de distancia e watershed.

O teste que mais importa aqui e o do teto de oraculo: se a gente alimentar o
watershed com a fronteira e a distancia perfeitas, o mAP tem que subir muito acima
do que componentes conexos consegue com a mascara semantica perfeita. Se esse teto
nao existisse, a Parte 2 inteira estaria atacando o problema errado e nem adiantaria
treinar.
"""

from __future__ import annotations

import numpy as np

from pa1.data.synthetic import generate_sample
from pa1.data.targets import (
    alpha_from_frequencies,
    class_frequencies,
    distance_map,
    semantic_mask,
    three_class_mask,
    touching_weight_map,
)
from pa1.metrics.instance import evaluate_image
from pa1.postprocess.naive import labels_from_probability
from pa1.postprocess.watershed import labels_from_boundary_head


def oracle_probs(labels, thickness=2):
    """As probabilidades que uma rede perfeita cuspiria."""
    three = three_class_mask(labels, thickness)
    return np.stack([three == 0, three == 1, three == 2]).astype(np.float32)


def test_boundary_separates_touching_instances():
    """A casca tem que aparecer no contato, senao a trilha A nao separa nada."""
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[10:30, 5:20] = 1
    labels[10:30, 20:35] = 2  # encostado no 1, sem nenhum pixel de fundo entre eles
    three = three_class_mask(labels, thickness=2)

    coluna = three[20, 15:25]
    assert (coluna == 2).any(), "sem fronteira no contato o watershed nao teria onde cortar"
    # e o interior das duas tem que ficar desconexo
    from scipy import ndimage as ndi

    n = ndi.label(three == 1, structure=np.ones((3, 3)))[1]
    assert n == 2, f"interior deveria dar 2 componentes, deu {n}"


def test_distance_map_peaks_at_the_center():
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[10:30, 10:30] = 1
    dist = distance_map(labels)
    assert dist.max() == 1.0
    assert dist[20, 20] > dist[11, 20] > 0
    assert dist[0, 0] == 0


def test_distance_does_not_leak_between_touching_instances():
    """Se a EDT fosse global, o vale entre dois objetos encostados sumiria."""
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[10:30, 5:20] = 1
    labels[10:30, 20:35] = 2
    dist = distance_map(labels)
    # na linha do meio tem que existir um vale no contato
    linha = dist[20, 5:35]
    assert linha[14] < linha[7], "a distancia tem que cair chegando no contato"
    assert linha[16] < linha[24]


def test_touching_weight_only_fires_between_different_instances():
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[10:30, 5:20] = 1
    labels[10:30, 20:35] = 2
    w = touching_weight_map(labels, w_touch=5.0, radius=3)
    assert w[20, 20] == 5.0            # no contato
    assert w[20, 7] == 1.0             # dentro do objeto 1, longe do contato
    assert w[2, 2] == 1.0              # fundo longe de tudo


def test_alpha_gives_more_weight_to_the_minority_class():
    labels = [generate_sample(np.random.default_rng(s))[1] for s in range(6)]
    freq = class_frequencies(labels, 2)
    assert freq[2] < freq[1] < freq[0], "fronteira tem que ser a classe minoritaria"
    alpha = alpha_from_frequencies(freq)
    assert alpha[2] > alpha[1] > alpha[0]


def test_watershed_beats_connected_components_with_perfect_input():
    """O teto de oraculo, que e o argumento inteiro da Parte 2."""
    cc_aps, ws_aps = [], []
    for seed in range(12):
        _, labels = generate_sample(np.random.default_rng(seed))
        cc = labels_from_probability(semantic_mask(labels), min_size=10)
        ws = labels_from_boundary_head(oracle_probs(labels), distance_map(labels), min_size=10)
        cc_aps.append(evaluate_image(cc, labels).ap)
        ws_aps.append(evaluate_image(ws, labels).ap)

    cc_map, ws_map = float(np.mean(cc_aps)), float(np.mean(ws_aps))
    assert cc_map < 0.25, f"o baseline tinha que fracassar, deu {cc_map:.3f}"
    assert ws_map > 0.70, f"o teto do watershed ficou baixo demais, deu {ws_map:.3f}"
    assert ws_map > 4 * cc_map


def test_watershed_falls_back_when_there_is_no_marker():
    """Sem interior previsto ele nao pode devolver imagem vazia."""
    probs = np.zeros((3, 30, 30), dtype=np.float32)
    probs[2, 5:25, 5:25] = 1.0  # so fronteira, nenhum interior
    probs[0] = 1.0 - probs[2]
    out = labels_from_boundary_head(probs, None, min_size=5)
    assert out.max() >= 1


def test_watershed_recovers_the_full_object_not_just_the_interior():
    """A mascara do watershed usa interior + fronteira, senao o IoU trava embaixo de 0.5."""
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[10:30, 10:30] = 1
    out = labels_from_boundary_head(oracle_probs(labels), distance_map(labels), min_size=5)
    assert evaluate_image(out, labels).ap > 0.9
