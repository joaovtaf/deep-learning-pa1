"""Checagem do breakdown de erro da Parte 5.

O caso do buraco na numeracao e o bug que derrubou o primeiro run no Kaggle: a area
saia com tamanho max(label) e a matriz de IoU com tamanho n_instancias, e as duas nao
alinhavam. So aparece quando algum id some do meio, que acontece depois de filtrar
instancia pequena.
"""

from __future__ import annotations

import numpy as np

from pa1.metrics.instance import error_breakdown, instance_areas


def test_merged_prediction_is_counted_as_merge():
    gt = np.zeros((20, 20), dtype=np.int32)
    gt[5:15, 2:11] = 1
    gt[5:15, 11:20] = 2
    merged = (gt > 0).astype(np.int32)
    b = error_breakdown(merged, gt)
    assert b["merged"] == 1
    assert b["fragmented"] == 0


def test_fragmented_prediction_is_counted_as_fragmentation():
    gt = np.zeros((20, 20), dtype=np.int32)
    gt[5:15, 2:18] = 1
    pred = np.zeros((20, 20), dtype=np.int32)
    pred[5:15, 2:10] = 1
    pred[5:15, 10:18] = 2
    b = error_breakdown(pred, gt)
    assert b["fragmented"] == 1
    assert b["merged"] == 0


def test_survives_a_gap_in_the_label_numbering():
    gt = np.zeros((20, 20), dtype=np.int32)
    gt[2:7, 2:7] = 1
    gt[12:17, 12:17] = 3  # o id 2 nao existe, que e o caso que quebrou antes
    pred = gt.copy()
    b = error_breakdown(pred, gt)
    assert b["n_pred"] == 2 and b["n_gt"] == 2
    assert b["missed"] == 0 and b["spurious"] == 0


def test_instance_areas_skips_missing_ids():
    lab = np.zeros((10, 10), dtype=np.int32)
    lab[0:2, 0:2] = 1   # area 4
    lab[5:8, 5:8] = 4   # area 9, ids 2 e 3 nao existem
    assert list(instance_areas(lab)) == [4, 9]


def test_empty_prediction():
    gt = np.zeros((10, 10), dtype=np.int32)
    gt[1:5, 1:5] = 1
    b = error_breakdown(np.zeros_like(gt), gt)
    assert b["missed"] == 1 and b["n_pred"] == 0
