"""Casos montados na mao pra checar o matching e o mAP.

O PA proibe usar AP de instancia de biblioteca, entao esses testes sao o que
garante que a nossa implementacao ta certa.
"""

from __future__ import annotations

import numpy as np
import pytest

from pa1.metrics.instance import (
    DEFAULT_THRESHOLDS,
    evaluate_image,
    instance_iou_matrix,
    match_instances,
    precision_at_threshold,
)


def two_squares(offset=0):
    """Canvas 20x20 com dois quadrados 5x5."""
    lab = np.zeros((20, 20), dtype=np.int32)
    lab[2:7, 2:7] = 1
    lab[12 + offset : 17 + offset, 12:17] = 2
    return lab


def test_iou_matrix_perfect_match():
    gt = two_squares()
    iou = instance_iou_matrix(gt, gt)
    assert iou.shape == (2, 2)
    assert np.allclose(np.diag(iou), 1.0)
    assert np.allclose(iou[0, 1], 0.0)


def test_iou_matrix_known_overlap():
    gt = np.zeros((10, 10), dtype=np.int32)
    gt[0:4, 0:4] = 1  # area 16
    pred = np.zeros((10, 10), dtype=np.int32)
    pred[0:4, 0:2] = 1  # area 8, todo dentro do gt
    iou = instance_iou_matrix(pred, gt)
    assert iou[0, 0] == pytest.approx(8 / 16)


def test_perfect_prediction_gives_map_one():
    gt = two_squares()
    res = evaluate_image(gt, gt)
    assert res.ap == pytest.approx(1.0)
    assert res.count_error == 0
    assert all(v == pytest.approx(1.0) for v in res.per_threshold.values())


def test_missing_instance_is_a_false_negative():
    gt = two_squares()
    pred = gt.copy()
    pred[pred == 2] = 0
    res = evaluate_image(pred, gt)
    # 1 TP, 0 FP, 1 FN em todo limiar, entao precisao 1/2
    assert res.ap == pytest.approx(0.5)
    assert res.n_pred == 1 and res.n_gt == 2
    assert res.count_error == 1


def test_spurious_instance_is_a_false_positive():
    gt = two_squares()
    pred = gt.copy()
    pred[0, 18] = 3  # objeto fantasma de 1 pixel
    res = evaluate_image(pred, gt)
    # 2 TP, 1 FP, 0 FN, entao 2/3
    assert res.ap == pytest.approx(2 / 3)
    assert res.count_error == 1


def test_precision_falls_as_the_threshold_rises():
    gt = np.zeros((20, 20), dtype=np.int32)
    gt[2:12, 2:12] = 1  # 100 px
    pred = np.zeros((20, 20), dtype=np.int32)
    pred[2:12, 2:10] = 1  # 80 px todo dentro, IoU 0.8
    iou = instance_iou_matrix(pred, gt)
    assert iou[0, 0] == pytest.approx(0.8)
    assert precision_at_threshold(iou, 0.50, 1, 1)[0] == pytest.approx(1.0)
    assert precision_at_threshold(iou, 0.85, 1, 1)[0] == pytest.approx(0.0)


def test_merged_instances_are_penalised():
    """O modo de falha da Parte 1: componentes conexos funde dois objetos encostados."""
    gt = np.zeros((20, 20), dtype=np.int32)
    gt[5:15, 2:11] = 1
    gt[5:15, 11:20] = 2
    merged = (gt > 0).astype(np.int32)
    res = evaluate_image(merged, gt)
    assert res.ap < 0.5
    assert res.n_pred == 1 and res.n_gt == 2


def test_greedy_and_hungarian_agree_on_an_unambiguous_case():
    gt = two_squares()
    iou = instance_iou_matrix(gt, gt)
    assert sorted(match_instances(iou, 0.5, "greedy")) == sorted(match_instances(iou, 0.5, "hungarian"))


def test_greedy_matching_is_one_to_one():
    """Duas predicoes em cima do mesmo GT, so a melhor pode casar."""
    gt = np.zeros((20, 20), dtype=np.int32)
    gt[0:10, 0:10] = 1
    pred = np.zeros((20, 20), dtype=np.int32)
    pred[0:9, 0:10] = 1   # IoU 0.9
    pred[9:10, 0:10] = 2  # IoU 0.1
    iou = instance_iou_matrix(pred, gt)
    pairs = match_instances(iou, 0.5, "greedy")
    assert len(pairs) == 1
    assert pairs[0][0] == 0


def test_empty_prediction_and_empty_ground_truth():
    empty = np.zeros((10, 10), dtype=np.int32)
    gt = two_squares()
    assert evaluate_image(empty, gt).ap == pytest.approx(0.0)
    assert evaluate_image(gt, empty).ap == pytest.approx(0.0)
    assert evaluate_image(empty, empty).ap == pytest.approx(1.0)


def test_threshold_grid_matches_the_assignment():
    assert DEFAULT_THRESHOLDS[0] == 0.50
    assert DEFAULT_THRESHOLDS[-1] == 0.95
    assert len(DEFAULT_THRESHOLDS) == 10
