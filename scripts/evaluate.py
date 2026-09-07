#!/usr/bin/env python
"""Comando que avalia.

Imprime IoU/Dice semanticos, mAP@[.50:.95], AP@.50 e erro medio de contagem, salva
um JSON com a tabela por imagem, o grafico de densidade do item 5 da Parte 1 e
alguns paineis qualitativos.

    uv run python scripts/evaluate.py --config configs/dsb2018_boundary.yaml \
        --checkpoint runs/dsb2018_boundary/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pa1.engine.datasets import build_datasets, build_loaders
from pa1.engine.evaluate import density_arrays, evaluate, format_report, save_metrics
from pa1.engine.train import build_from_config
from pa1.utils import get_device, load_checkpoint, load_config, seed_everything
from pa1.viz import density_plot, panel


@torch.no_grad()
def qualitative_panels(model, dataset, task, device, out_dir: Path, n=4):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for i in range(min(n, len(dataset))):
        item = dataset[i]
        logits = model(item["image"].unsqueeze(0).to(device))
        pred = task.decode(logits)[0]
        maps = {k: v[0] for k, v in task.maps(logits).items()}
        raw_image, gt_labels = dataset.raw(i)
        h, w = gt_labels.shape
        panel(
            raw_image,
            gt_labels,
            pred[:h, :w],
            {k: v[:h, :w] for k, v in maps.items()},
            title=f"{dataset.image_ids[i][:12]}",
            path=out_dir / f"sample_{i}.png",
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--matching", default=None, choices=["greedy", "hungarian"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--panels", type=int, default=4)
    # sobrescrevem o bloco postprocess: do config, pra medir o efeito da calibracao sem
    # ter que criar um yaml novo pra cada combinacao
    ap.add_argument("--interior-threshold", type=float, default=None)
    ap.add_argument("--fg-threshold", type=float, default=None)
    ap.add_argument("--min-size", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    # fg_threshold e threshold sao a mesma ideia nas duas tarefas (ate onde o objeto
    # cresce), so mudam de nome, entao um flag so escreve os dois
    overrides = [("interior_threshold", args.interior_threshold), ("min_size", args.min_size),
                 ("fg_threshold", args.fg_threshold), ("threshold", args.fg_threshold)]
    for key, value in overrides:
        if value is not None:
            cfg.setdefault("postprocess", {})[key] = value
    seed_everything(int(cfg.get("seed", 0)))
    device = get_device(args.device or cfg.get("device", "auto"))

    datasets = build_datasets(cfg["data"])
    loaders = build_loaders(datasets, cfg.get("loader", {}))
    model, task = build_from_config(cfg, device)
    load_checkpoint(args.checkpoint, model, map_location=device)

    matching = args.matching or cfg.get("matching", "greedy")
    metrics = evaluate(model, loaders[args.split], device, task, matching=matching)
    print(format_report(metrics, f"{cfg.get('name', args.config)}, split {args.split}"))

    out_dir = Path(args.out_dir or Path(args.checkpoint).parent / f"eval_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, out_dir / "metrics.json")

    d, ap_, ce = density_arrays(metrics)
    density_plot(d, ap_, ce, path=out_dir / "density.png",
                 title=f"{cfg.get('name','')}: mAP e erro de contagem contra densidade")
    qualitative_panels(model, datasets[args.split], task, device, out_dir / "panels", n=args.panels)
    print(f"\nsalvou metrics.json, density.png e {args.panels} paineis em {out_dir}")


if __name__ == "__main__":
    main()
