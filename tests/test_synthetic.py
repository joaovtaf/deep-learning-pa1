"""Checagens do dataset sintetico da Parte 0 e do decoder ingenuo."""

from __future__ import annotations

import numpy as np

from pa1.data.synthetic import SyntheticEllipsesDataset, generate_sample
from pa1.data.targets import distance_map, semantic_mask, three_class_mask
from pa1.metrics.instance import evaluate_image
from pa1.postprocess.naive import labels_from_probability


def test_sample_shapes_and_range():
    image, labels = generate_sample(np.random.default_rng(0))
    assert image.shape == (128, 128) and labels.shape == (128, 128)
    assert image.dtype == np.float32 and 0.0 <= image.min() and image.max() <= 1.0
    assert labels.min() == 0


def test_instance_count_inside_the_requested_range():
    for seed in range(20):
        _, labels = generate_sample(np.random.default_rng(seed))
        n = int(labels.max())
        assert 5 <= n <= 20, f"seed {seed} gerou {n} instancias"
        assert set(np.unique(labels)) == set(range(n + 1)), "labels tem que ser 1..n sem buraco"


def test_dataset_is_deterministic():
    a = SyntheticEllipsesDataset(length=4, seed=7)
    b = SyntheticEllipsesDataset(length=4, seed=7)
    ia, la = a.raw(2)
    ib, lb = b.raw(2)
    assert np.array_equal(ia, ib) and np.array_equal(la, lb)


def test_dataset_item_tensor_shapes():
    ds = SyntheticEllipsesDataset(length=2, seed=0)
    item = ds[0]
    assert item["image"].shape == (3, 128, 128)
    assert item["mask"].shape == (1, 128, 128)
    assert item["labels"].shape == (128, 128)
    assert set(np.unique(item["mask"].numpy())) <= {0.0, 1.0}


def test_many_instances_actually_touch():
    """O ponto do dataset: mesmo com mascara semantica perfeita o CC funde as elipses."""
    merged = 0
    oracle_aps = []
    for seed in range(12):
        _, labels = generate_sample(np.random.default_rng(seed))
        cc = labels_from_probability(semantic_mask(labels))
        oracle_aps.append(evaluate_image(cc, labels).ap)
        merged += int(cc.max() < labels.max())
    assert merged >= 9
    assert np.mean(oracle_aps) < 0.6


def test_target_derivations():
    _, labels = generate_sample(np.random.default_rng(3))
    three = three_class_mask(labels, thickness=2)
    assert set(np.unique(three)) <= {0, 1, 2}
    assert (three == 2).sum() > 0, "a classe fronteira nao pode ficar vazia"
    # fronteira e minoritaria, que e o desbalanceamento que o eixo 2 da Parte 3 ataca
    assert (three == 2).mean() < (three == 1).mean()

    dist = distance_map(labels)
    assert dist.shape == labels.shape
    assert dist.max() <= 1.0 + 1e-6
    assert np.all(dist[labels == 0] == 0)
