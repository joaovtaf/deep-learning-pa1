"""As duas tarefas do PA, com a mesma interface.

"binary" e a Parte 1: 1 canal de foreground, BCE+Dice, instancias por limiar +
componentes conexos.

"boundary" e a Parte 2 trilha A: 4 canais, sendo 3 de classe (fundo / interior /
fronteira) e 1 de distancia ao fundo. A perda de classe e CE, CE balanceada ou focal
(slides 73 a 79) e a de distancia e L1 ou L2 (slide 80). A decodificacao e watershed
com os interiores como marcador.

Manter as duas atras da mesma interface e o que deixa o resto do codigo (loop de
treino, avaliacao, tiling da Parte 4, corrupcoes da Parte 6) nao saber qual das duas
esta rodando.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..losses import build_loss
from ..models.unet import split_boundary_output
from ..postprocess.naive import labels_from_probability
from ..postprocess.watershed import labels_from_boundary_head


class BinaryTask:
    name = "binary"
    out_channels = 1

    def __init__(self, loss_cfg: dict, postprocess: dict | None = None):
        self.criterion = build_loss(loss_cfg)
        self.pp = dict(postprocess or {})

    def loss(self, logits, batch, device):
        return self.criterion(logits, batch["mask"].to(device))

    @staticmethod
    def foreground_prob(logits) -> np.ndarray:
        return torch.sigmoid(logits)[:, 0].detach().cpu().numpy()

    def decode(self, logits) -> list[np.ndarray]:
        prob = self.foreground_prob(logits)
        return [
            labels_from_probability(
                p, threshold=self.pp.get("threshold", 0.5), min_size=self.pp.get("min_size", 20)
            )
            for p in prob
        ]

    def maps(self, logits) -> dict:
        """Mapas intermediarios pras figuras da Parte 5."""
        return {"foreground": self.foreground_prob(logits)}


class BoundaryTask:
    name = "boundary"
    out_channels = 4

    def __init__(
        self,
        loss_cfg: dict,
        distance_loss_cfg: dict | None = None,
        distance_weight: float = 1.0,
        touching_weight: float = 0.0,
        postprocess: dict | None = None,
    ):
        self.criterion = build_loss(loss_cfg)
        self.dist_criterion = build_loss(distance_loss_cfg or {"name": "regression", "kind": "l1"})
        self.distance_weight = float(distance_weight)
        self.touching_weight = float(touching_weight)
        self.pp = dict(postprocess or {})

    def loss(self, logits, batch, device):
        class_logits, dist_pred = split_boundary_output(logits)
        three = batch["three"].to(device)
        dist = batch["dist"].to(device)

        weight = None
        if self.touching_weight > 0 and "touch_weight" in batch:
            weight = batch["touch_weight"].to(device)

        cls = self.criterion(class_logits, three, weight=weight)
        # a regressao so faz sentido no foreground, no fundo o alvo e zero constante
        fg = (three > 0).unsqueeze(1).float()
        reg = self.dist_criterion(dist_pred, dist, mask=fg)
        return cls + self.distance_weight * reg

    @staticmethod
    def _split(logits):
        class_logits, dist_pred = split_boundary_output(logits)
        probs = torch.softmax(class_logits, dim=1).detach().cpu().numpy()
        return probs, dist_pred[:, 0].detach().cpu().numpy()

    @staticmethod
    def foreground_prob(logits) -> np.ndarray:
        probs = torch.softmax(split_boundary_output(logits)[0], dim=1)
        return (probs[:, 1] + probs[:, 2]).detach().cpu().numpy()

    def decode(self, logits) -> list[np.ndarray]:
        probs, dist = self._split(logits)
        return [
            labels_from_boundary_head(
                probs[b],
                dist[b],
                interior_threshold=self.pp.get("interior_threshold", 0.5),
                fg_threshold=self.pp.get("fg_threshold", 0.5),
                min_size=self.pp.get("min_size", 20),
                min_marker_size=self.pp.get("min_marker_size", 4),
            )
            for b in range(probs.shape[0])
        ]

    def maps(self, logits) -> dict:
        probs, dist = self._split(logits)
        return {
            "foreground": probs[:, 1] + probs[:, 2],
            "interior": probs[:, 1],
            "boundary": probs[:, 2],
            "distance": dist,
        }


def build_task(cfg: dict):
    cfg = dict(cfg)
    name = cfg.pop("name", "binary").lower()
    if name == "binary":
        return BinaryTask(**cfg)
    if name == "boundary":
        return BoundaryTask(**cfg)
    raise KeyError(f"tarefa '{name}' nao existe, tem 'binary' e 'boundary'")
