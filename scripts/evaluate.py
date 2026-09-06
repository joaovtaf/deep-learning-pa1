#!/usr/bin/env python
"""Comando que avalia.

Imprime IoU/Dice semanticos, mAP@[.50:.95], AP@.50 e erro medio de contagem,
salva um JSON com a tabela por imagem, o grafico de densidade do item 5 da
Parte 1 e alguns paineis qualitativos.

    uv run python scripts/evaluate.py --config configs/dsb2018_baseline.yaml \
        --checkpoint runs/dsb2018_baseline/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from pa1.engine.datasets import build_datasets, build_loaders
from pa1.engine.evaluate import density_arrays, evaluate, format_report, save_metrics
from pa1.models.unet import build_model
from pa1.postprocess.naive import labels_from_probability
from pa1.utils import get_device, load_checkpoint, load_config, seed_everything
from pa1.viz import density_plot, panel


@torch.no_grad()
def qualitative_panels(model, dataset, device, out_dir: Path, n=4, threshold=0.5, min_size=10):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for i in range(min(n, len(dataset))):
        item = dataset[i]
        image = item["image"].unsqueeze(0).to(device)
        prob = torch.sigmoid(model(image))[0, 0].cpu().numpy()
        pred = labels_from_probability(prob, threshold=threshold, min_size=min_size)
        raw_image, gt_labels = dataset.raw(i)
        if raw_image.dtype == np.uint8:
            raw_image = raw_image / 255.0
        # a predicao vem com padding, corta de volta pro tamanho original
        h, w = gt_labels.shape
        panel(raw_image, gt_labels, pred[:h, :w], prob[:h, :w], title=f"amostra {i}", path=out_dir / f"sample_{i}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--matching", default=None, choices=["greedy", "hungarian"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--panels", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 0)))
    device = get_device(args.device or cfg.get("device", "auto"))

    datasets = build_datasets(cfg["data"])
    loaders = build_loaders(datasets, cfg.get("loader", {}))

    model = build_model(cfg["model"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)

    pp = cfg.get("postprocess", {})
    threshold = float(pp.get("threshold", 0.5))
    min_size = int(pp.get("min_size", 10))
    matching = args.matching or cfg.get("matching", "greedy")

    metrics = evaluate(model, loaders[args.split], device, threshold=threshold, min_size=min_size, matching=matching)
    print(format_report(metrics, f"{cfg.get('name', args.config)}, split {args.split}"))

    out_dir = Path(args.out_dir or Path(args.checkpoint).parent / f"eval_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, out_dir / "metrics.json")

    d, ap_, ce = density_arrays(metrics)
    density_plot(d, ap_, ce, path=out_dir / "density.png")
    qualitative_panels(
        model, datasets[args.split], device, out_dir / "panels", n=args.panels, threshold=threshold, min_size=min_size
    )
    print(f"\nsalvou metrics.json, density.png e {args.panels} paineis em {out_dir}")


if __name__ == "__main__":
    main()
