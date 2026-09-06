"""Losses escritas na mao.

A aula deu CE, CE balanceada, focal e L1/L2 (slides 73 a 80). Como a Parte 3
eixo 2 pede pra variar gamma e alpha, deixei tudo com a mesma interface e com um
weight opcional por pixel, que a gente vai precisar na Parte 2 pra dar mais peso
pra classe fronteira.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ---------- binario (baseline da Parte 1) ----------
class BCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight: float | None = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        pw = None
        if self.pos_weight is not None:
            pw = torch.as_tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, target.float(), pos_weight=pw)


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        probs = torch.sigmoid(logits).flatten(1)
        target = target.float().flatten(1)
        inter = (probs * target).sum(1)
        union = probs.sum(1) + target.sum(1)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """BCE + Dice, que e o que a gente usa no baseline."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, pos_weight: float | None = None):
        super().__init__()
        self.bce = BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return self.bce_weight * self.bce(logits, target) + self.dice_weight * self.dice(logits, target)


# ---------- multiclasse (cabeca de fronteira da Parte 2) ----------
class CrossEntropyLoss(nn.Module):
    """CE normal, ou balanceada se passar alpha (vetor de peso por classe)."""

    def __init__(self, alpha: list[float] | Tensor | None = None, ignore_index: int = -100):
        super().__init__()
        self.register_buffer(
            "alpha",
            torch.as_tensor(alpha, dtype=torch.float32) if alpha is not None else None,
            persistent=False,
        )
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, target: Tensor, weight: Tensor | None = None) -> Tensor:
        alpha = self.alpha.to(logits.device) if self.alpha is not None else None
        ce = F.cross_entropy(
            logits, target.long(), weight=alpha, ignore_index=self.ignore_index, reduction="none"
        )
        if weight is not None:
            ce = ce * weight
        return ce.mean()


class FocalLoss(nn.Module):
    """Focal multiclasse. gamma=0 vira CE, e com alpha vira focal balanceada."""

    def __init__(self, gamma: float = 2.0, alpha: list[float] | Tensor | None = None, ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.register_buffer(
            "alpha",
            torch.as_tensor(alpha, dtype=torch.float32) if alpha is not None else None,
            persistent=False,
        )
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, target: Tensor, weight: Tensor | None = None) -> Tensor:
        target = target.long()
        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()

        valid = target != self.ignore_index
        safe_target = target.clone()
        safe_target[~valid] = 0

        idx = safe_target.unsqueeze(1)
        log_pt = log_prob.gather(1, idx).squeeze(1)
        pt = prob.gather(1, idx).squeeze(1)

        loss = -((1.0 - pt) ** self.gamma) * log_pt

        if self.alpha is not None:
            at = self.alpha.to(logits.device).gather(0, safe_target.flatten()).view_as(loss)
            loss = loss * at
        if weight is not None:
            loss = loss * weight

        loss = loss * valid
        return loss.sum() / valid.sum().clamp_min(1)


# ---------- regressao (mapa de distancia, slide 80) ----------
class MaskedRegressionLoss(nn.Module):
    """L1 ou L2, opcionalmente so dentro de uma mascara (tipo so no foreground)."""

    def __init__(self, kind: str = "l1"):
        super().__init__()
        assert kind in {"l1", "l2"}
        self.kind = kind

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
        diff = pred - target
        err = diff.abs() if self.kind == "l1" else diff.pow(2)
        if mask is not None:
            err = err * mask
            return err.sum() / mask.sum().clamp_min(1)
        return err.mean()


LOSSES = {
    "bce": BCEWithLogitsLoss,
    "dice": DiceLoss,
    "bce_dice": BCEDiceLoss,
    "ce": CrossEntropyLoss,
    "focal": FocalLoss,
    "regression": MaskedRegressionLoss,
}


def build_loss(cfg: dict) -> nn.Module:
    cfg = dict(cfg)
    name = cfg.pop("name").lower()
    if name not in LOSSES:
        raise KeyError(f"loss '{name}' nao existe, tem {sorted(LOSSES)}")
    return LOSSES[name](**cfg)
