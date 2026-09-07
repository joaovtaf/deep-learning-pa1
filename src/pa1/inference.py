"""Inferencia numa imagem solta, que e o que o inferencia.ipynb usa.

Recebe o caminho de uma imagem qualquer, devolve o mapa de instancias e a contagem.
Nao retreina nada, so carrega o checkpoint.

O tiling entra automatico quando a imagem e maior que max_size. A escolha default e
o modo blend (media dos logits e uma decodificacao so no fim), que e o que a Parte 4
mostrou ser o melhor dos tres, e nao o per_tile do slide 83 aplicado ao pe da letra.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .engine.train import build_from_config
from .tiling import blend_logits, predict_tiles
from .utils import as_rgb_uint8, get_device, load_checkpoint, load_config
from .viz import colorize_labels, overlay_instances


def read_any_image(path) -> np.ndarray:
    """Abre PNG/JPG/TIF e devolve RGB uint8. O DSB2018 salva RGBA e o alfa nao serve."""
    return np.array(Image.open(Path(path)).convert("RGB"))


class Predictor:
    def __init__(self, config, checkpoint, device=None, max_size=768, tile=256, overlap=64):
        self.cfg = load_config(config)
        self.device = get_device(device or self.cfg.get("device", "auto"))
        self.model, self.task = build_from_config(self.cfg, self.device)
        self.meta = load_checkpoint(checkpoint, self.model, map_location=self.device)
        self.model.eval()
        self.max_size = int(max_size)
        self.tile = int(tile)
        self.overlap = int(overlap)

    @torch.no_grad()
    def _logits_full(self, image):
        h, w = image.shape[:2]
        ph, pw = (-h) % 32, (-w) % 32
        padded = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        x = torch.from_numpy(padded.astype(np.float32).transpose(2, 0, 1)[None] / 255.0).to(self.device)
        return self.model(x)[:, :, :h, :w].cpu()

    @torch.no_grad()
    def _logits_tiled(self, image):
        logits, coords = predict_tiles(self.model, image, self.device, self.tile, self.overlap)
        return blend_logits(logits, coords, image.shape[:2], self.tile, self.overlap).unsqueeze(0)

    def predict(self, path_or_array) -> dict:
        image = read_any_image(path_or_array) if isinstance(path_or_array, (str, Path)) \
            else as_rgb_uint8(path_or_array)
        h, w = image.shape[:2]
        tiled = max(h, w) > self.max_size
        logits = self._logits_tiled(image) if tiled else self._logits_full(image)

        labels = self.task.decode(logits)[0][:h, :w]
        maps = {k: v[0][:h, :w] for k, v in self.task.maps(logits).items()}
        return {
            "image": image,
            "labels": labels,
            "count": int(labels.max()),
            "colored": colorize_labels(labels),
            "overlay": overlay_instances(image, labels),
            "maps": maps,
            "tiled": tiled,
        }


def load_predictor(config="configs/dsb2018_boundary.yaml", checkpoint="runs/dsb2018_boundary/best.pt", **kw):
    return Predictor(config, checkpoint, **kw)
