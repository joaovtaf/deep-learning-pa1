"""IoU e Dice binarios, que sao os numeros semanticos que a Parte 1 pede."""

from __future__ import annotations

import numpy as np


def _binarize(x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(x) > threshold).astype(bool)


def iou_score(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5, eps: float = 1e-7) -> float:
    p = _binarize(pred, threshold)
    t = _binarize(target, 0.5)
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    return float((inter + eps) / (union + eps))


def dice_score(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5, eps: float = 1e-7) -> float:
    p = _binarize(pred, threshold)
    t = _binarize(target, 0.5)
    inter = np.logical_and(p, t).sum()
    return float((2 * inter + eps) / (p.sum() + t.sum() + eps))


class SemanticMeter:
    """Media dos scores por imagem ao longo do dataset."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._iou: list[float] = []
        self._dice: list[float] = []

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        self._iou.append(iou_score(pred, target, self.threshold))
        self._dice.append(dice_score(pred, target, self.threshold))

    def compute(self) -> dict[str, float]:
        return {
            "iou": float(np.mean(self._iou)) if self._iou else 0.0,
            "dice": float(np.mean(self._dice)) if self._dice else 0.0,
            "n": len(self._iou),
        }
