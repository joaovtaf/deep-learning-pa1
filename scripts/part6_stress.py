#!/usr/bin/env python
"""Parte 6: teste de estresse. Escolhemos a opcao das corrupcoes.

Blur, ruido e brilho/contraste, cada um em 3 intensidades, aplicados so na avaliacao.
Sai a curva de degradacao do mAP e a tabela com IoU/Dice do lado, que e o que mostra
se o modelo esta perdendo a capacidade de achar o objeto (Dice cai) ou so a de separar
objetos encostados (Dice segura e mAP cai).

    uv run python scripts/part6_stress.py --config configs/dsb2018_boundary.yaml \
        --checkpoint runs/dsb2018_boundary/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pa1.corruptions import CORRUPTIONS, SEVERITIES, apply_corruption
from pa1.engine.datasets import build_datasets
from pa1.engine.train import build_from_config
from pa1.metrics.instance import InstanceMeter
from pa1.metrics.semantic import SemanticMeter
from pa1.utils import as_rgb_uint8, get_device, load_checkpoint, load_config, seed_everything
from pa1.viz import degradation_curve, panel


@torch.no_grad()
def evaluate_corrupted(model, task, dataset, device, name, severity, seed=0):
    sem, inst = SemanticMeter(0.5), InstanceMeter()
    rng = np.random.default_rng(seed)
    for i in range(len(dataset)):
        image, gt = dataset.raw(i)
        image = apply_corruption(as_rgb_uint8(image), name, severity, rng)

        h, w = gt.shape
        ph, pw = (-h) % 32, (-w) % 32
        padded = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        x = torch.from_numpy(padded.astype(np.float32).transpose(2, 0, 1)[None] / 255.0).to(device)

        logits = model(x)
        pred = task.decode(logits)[0][:h, :w]
        fg = task.foreground_prob(logits)[0][:h, :w]
        sem.update(fg, gt > 0)
        inst.update(pred, gt)
    return {**sem.compute(), **inst.compute()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 0)))
    device = get_device(args.device or cfg.get("device", "auto"))

    datasets = build_datasets(cfg["data"])
    dataset = datasets[args.split]
    model, task = build_from_config(cfg, device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    out_dir = Path(args.out_dir or Path(args.checkpoint).parent / "part6_stress")
    out_dir.mkdir(parents=True, exist_ok=True)

    clean = evaluate_corrupted(model, task, dataset, device, "none", 0)
    print(f"limpo: mAP {clean['mAP']:.4f}  Dice {clean['dice']:.4f}  "
          f"|err contagem| {clean['mean_count_error']:.2f}")

    table = {}
    rows = [{"corruption": "none", "severity": 0, **{k: clean[k] for k in
             ("mAP", "AP50", "iou", "dice", "mean_count_error")}}]
    for name in CORRUPTIONS:
        sev, vals = [0], [clean["mAP"]]
        for s in SEVERITIES:
            m = evaluate_corrupted(model, task, dataset, device, name, s)
            sev.append(s)
            vals.append(m["mAP"])
            rows.append({"corruption": name, "severity": s,
                         **{k: m[k] for k in ("mAP", "AP50", "iou", "dice", "mean_count_error")}})
            print(f"  {name:<20} s{s}  mAP {m['mAP']:.4f}  Dice {m['dice']:.4f}  "
                  f"|err contagem| {m['mean_count_error']:.2f}")
        table[name] = (sev, vals)

    degradation_curve(table, path=out_dir / "degradation.png")

    # figura qualitativa da corrupcao mais forte de cada familia
    image, gt = as_rgb_uint8(dataset.raw(0)[0]), dataset.raw(0)[1]
    for name in CORRUPTIONS:
        corrupted = apply_corruption(image, name, 3)
        h, w = gt.shape
        ph, pw = (-h) % 32, (-w) % 32
        x = torch.from_numpy(np.pad(corrupted, ((0, ph), (0, pw), (0, 0)), mode="reflect")
                             .astype(np.float32).transpose(2, 0, 1)[None] / 255.0).to(device)
        with torch.no_grad():
            logits = model(x)
        pred = task.decode(logits)[0][:h, :w]
        maps = {k: v[0][:h, :w] for k, v in task.maps(logits).items()}
        panel(corrupted, gt, pred, maps, title=f"{name} intensidade 3", path=out_dir / f"{name}_s3.png")

    (out_dir / "results.json").write_text(json.dumps({"clean": {k: v for k, v in clean.items()
                                                                if not k.startswith("_")},
                                                      "rows": rows}, indent=2, default=float))
    print(f"\nsalvou degradation.png e results.json em {out_dir}")


if __name__ == "__main__":
    main()
