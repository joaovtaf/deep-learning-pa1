"""Funcoes soltas que todo mundo usa: seed, device, config, checkpoint."""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def tqdm_disabled() -> bool:
    # barra de progresso so quando ta rodando no terminal, senao polui o log
    return not sys.stderr.isatty()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: str | None = None) -> torch.device:
    """cuda se tiver, senao mps (mac), senao cpu."""
    if prefer and prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg or {}


def save_checkpoint(path: str | Path, model: torch.nn.Module, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "meta": meta or {}}, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, map_location: str | torch.device = "cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return ckpt.get("meta", {})


@dataclass
class AverageMeter:
    """Media corrente, pra loss do treino."""

    total: float = 0.0
    count: int = 0
    history: list[float] = field(default_factory=list)

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n
        self.history.append(float(value))

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


def as_rgb_uint8(image) -> np.ndarray:
    """Normaliza o que sai de dataset.raw().

    O sintetico devolve float 2D em [0,1] e o DSB2018 devolve uint8 RGB. Tudo que
    consome imagem crua (tiling da Parte 4, corrupcoes da Parte 6) quer o mesmo
    formato, entao a conversao mora num lugar so.
    """
    x = np.asarray(image)
    if x.dtype != np.uint8:
        x = np.clip(x * 255.0 if x.max() <= 1.0 else x, 0, 255).astype(np.uint8)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    return x
