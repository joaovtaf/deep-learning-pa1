#!/usr/bin/env python
"""Parte 2: as mesmas metricas lado a lado com a baseline da Parte 1.

Avalia dois checkpoints no mesmo split e monta a tabela comparativa, a curva de
densidade sobreposta e paineis das imagens onde a diferenca e maior. E o slide
central da apresentacao.

    uv run python scripts/compare.py \
        --baseline configs/dsb2018_baseline.yaml runs/dsb2018_baseline/best.pt \
        --instance configs/dsb2018_boundary.yaml runs/dsb2018_boundary/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pa1.engine.datasets import build_datasets, build_loaders
from pa1.engine.evaluate import density_arrays, evaluate, format_report
from pa1.engine.train import build_from_config
from pa1.utils import get_device, load_checkpoint, load_config, seed_everything
from pa1.viz import density_comparison, panel


def load(config_path, ckpt_path, device):
    cfg = load_config(config_path)
    model, task = build_from_config(cfg, device)
    load_checkpoint(ckpt_path, model, map_location=device)
    model.eval()
    return cfg, model, task


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", nargs=2, required=True, metavar=("CONFIG", "CKPT"))
    ap.add_argument("--instance", nargs=2, required=True, metavar=("CONFIG", "CKPT"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--matching", default="greedy", choices=["greedy", "hungarian"])
    ap.add_argument("--out-dir", default="runs/comparison")
    ap.add_argument("--panels", type=int, default=4)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = get_device(args.device or "auto")
    seed_everything(0)

    cfg_b, model_b, task_b = load(*args.baseline, device)
    cfg_i, model_i, task_i = load(*args.instance, device)

    datasets = build_datasets(cfg_i["data"])
    loaders = build_loaders(datasets, cfg_i.get("loader", {}))
    dataset = datasets[args.split]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    series = {}
    for label, model, task in (("Parte 1 (limiar + CC)", model_b, task_b),
                               ("Parte 2 (fronteira + watershed)", model_i, task_i)):
        m = evaluate(model, loaders[args.split], device, task, matching=args.matching)
        print(format_report(m, f"{label}, split {args.split}"))
        print()
        results[label] = {k: v for k, v in m.items() if not k.startswith("_")}
        series[label] = density_arrays(m)

    density_comparison(series, path=out_dir / "density_comparison.png",
                       title="Parte 1 contra Parte 2: o que muda com a densidade")

    # imagens onde a Parte 2 mais ganha da Parte 1
    from pa1.metrics.instance import evaluate_image

    deltas = []
    for i in range(len(dataset)):
        item = dataset[i]
        x = item["image"].unsqueeze(0).to(device)
        _, gt = dataset.raw(i)
        h, w = gt.shape
        lb = task_b.decode(model_b(x))[0][:h, :w]
        li = task_i.decode(model_i(x))[0][:h, :w]
        ab, ai = evaluate_image(lb, gt).ap, evaluate_image(li, gt).ap
        deltas.append((ai - ab, i, ab, ai))
    deltas.sort(reverse=True)

    for rank, (delta, i, ab, ai) in enumerate(deltas[: args.panels], start=1):
        item = dataset[i]
        x = item["image"].unsqueeze(0).to(device)
        image, gt = dataset.raw(i)
        h, w = gt.shape
        lb = task_b.decode(model_b(x))[0][:h, :w]
        logits_i = model_i(x)
        li = task_i.decode(logits_i)[0][:h, :w]
        maps = {k: v[0][:h, :w] for k, v in task_i.maps(logits_i).items() if k in ("interior", "boundary")}
        panel(image, gt, lb, {}, title=f"{rank}. Parte 1, AP {ab:.3f}", path=out_dir / f"gain_{rank}_parte1.png")
        panel(image, gt, li, maps, title=f"{rank}. Parte 2, AP {ai:.3f} (ganho {delta:+.3f})",
              path=out_dir / f"gain_{rank}_parte2.png")

    table = {"split": args.split, "matching": args.matching, "results": results,
             "top_gains": [{"idx": i, "image_id": dataset.image_ids[i], "ap_baseline": ab,
                            "ap_instance": ai, "delta": d} for d, i, ab, ai in deltas[:10]]}
    (out_dir / "comparison.json").write_text(json.dumps(table, indent=2, default=float))

    b = results["Parte 1 (limiar + CC)"]
    n = results["Parte 2 (fronteira + watershed)"]
    print(f"{'metrica':<20}{'Parte 1':>12}{'Parte 2':>12}{'delta':>12}")
    for k in ("iou", "dice", "mAP", "AP50", "mean_count_error"):
        print(f"{k:<20}{b[k]:>12.4f}{n[k]:>12.4f}{n[k]-b[k]:>+12.4f}")
    print(f"\nsalvou a comparacao em {out_dir}")


if __name__ == "__main__":
    main()
