#!/usr/bin/env python
"""Varre os hiperparametros da decodificacao no split de validacao.

Serve pras duas tarefas. Na binaria da Parte 1 varre limiar e min_size, na trilha A varre
interior_threshold, fg_threshold e min_size. Calibrar as duas com o mesmo criterio importa
pra comparacao da Parte 2 ser justa: nao adianta calibrar so o modelo novo e comparar com
o baseline no chute.

O watershed tem tres numeros que a gente escolheu no chute e que mudam bastante o mAP
sem mexer na rede. Escolher eles na validacao e nao no teste e o minimo de higiene.

interior_threshold controla quantos marcadores nascem. Alto demais mata nucleo pequeno
(vira falso negativo), baixo demais deixa dois marcadores dentro do mesmo nucleo e ele
racha (vira falso positivo). E o que mais importa dos tres.

fg_threshold controla ate onde o objeto cresce, ou seja, mexe no IoU de cada instancia
casada, nao no numero delas.

min_size joga fora instancia pequena demais.

Roda so inferencia, entao e barato: uma passada pela rede e depois so numpy.

    uv run python scripts/tune_watershed.py --config configs/dsb2018_boundary.yaml \
        --checkpoint runs/dsb2018_boundary/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pa1.engine.datasets import build_datasets
from pa1.engine.train import build_from_config
from pa1.metrics.instance import InstanceMeter
from pa1.postprocess.naive import labels_from_probability
from pa1.postprocess.watershed import labels_from_boundary_head
from pa1.utils import get_device, load_checkpoint, load_config, seed_everything


@torch.no_grad()
def cache_predictions(model, task, dataset, device):
    """Roda a rede uma vez so e guarda a saida, pra varrer os limiares em cima em numpy."""
    cached = []
    for i in range(len(dataset)):
        item = dataset[i]
        logits = model(item["image"].unsqueeze(0).to(device))
        _, gt = dataset.raw(i)
        h, w = gt.shape
        if task.name == "boundary":
            probs, dist = task._split(logits)
            cached.append((probs[0][:, :h, :w], dist[0][:h, :w], gt))
        else:
            cached.append((task.foreground_prob(logits)[0][:h, :w], None, gt))
    return cached


def sweep_binary(cached, thresholds, min_sizes):
    rows = []
    print(f"\n{'limiar':>10}{'min_size':>10}{'mAP':>10}{'AP50':>10}{'errCont':>10}")
    for th in thresholds:
        for ms in min_sizes:
            meter = InstanceMeter()
            for prob, _, gt in cached:
                meter.update(labels_from_probability(prob, threshold=th, min_size=ms), gt)
            m = meter.compute()
            rows.append({"threshold": th, "min_size": ms, "mAP": m["mAP"], "AP50": m["AP50"],
                         "mean_count_error": m["mean_count_error"]})
            print(f"{th:>10.2f}{ms:>10}{m['mAP']:>10.4f}{m['AP50']:>10.4f}"
                  f"{m['mean_count_error']:>10.2f}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--interior", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    # o fg_threshold sobe ate 0.9 porque na primeira varredura (que parava em 0.6) o
    # melhor caiu bem na borda, o que quase sempre quer dizer que o otimo ta la fora
    ap.add_argument("--foreground", type=float, nargs="+",
                    default=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--min-size", type=int, nargs="+", default=[10, 20, 40, 80])
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 0)))
    device = get_device(args.device or cfg.get("device", "auto"))

    dataset = build_datasets(cfg["data"])[args.split]
    model, task = build_from_config(cfg, device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    print(f"rodando a rede em {len(dataset)} imagens do split {args.split}")
    cached = cache_predictions(model, task, dataset, device)

    if task.name == "binary":
        # na Parte 1 so existem dois botoes, o limiar da probabilidade e o min_size
        rows = sweep_binary(cached, args.foreground, args.min_size)
        best = max(rows, key=lambda r: r["mAP"])
        print(f"\nmelhor no split {args.split}: limiar {best['threshold']}, "
              f"min_size {best['min_size']}  ->  mAP {best['mAP']:.4f}")
        out = Path(args.out or Path(args.checkpoint).parent / "tune_postprocess.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"split": args.split, "task": "binary", "best": best,
                                   "grid": rows}, indent=2, default=float))
        print(f"salvou em {out}")
        return

    rows = []
    print(f"\n{'interior':>10}{'foreground':>12}{'min_size':>10}{'mAP':>10}{'AP50':>10}{'errCont':>10}")
    for it in args.interior:
        for fg in args.foreground:
            for ms in args.min_size:
                meter = InstanceMeter()
                for probs, dist, gt in cached:
                    pred = labels_from_boundary_head(probs, dist, interior_threshold=it,
                                                     fg_threshold=fg, min_size=ms)
                    meter.update(pred, gt)
                m = meter.compute()
                rows.append({"interior_threshold": it, "fg_threshold": fg, "min_size": ms,
                             "mAP": m["mAP"], "AP50": m["AP50"],
                             "mean_count_error": m["mean_count_error"]})
                print(f"{it:>10.2f}{fg:>12.2f}{ms:>10}{m['mAP']:>10.4f}{m['AP50']:>10.4f}"
                      f"{m['mean_count_error']:>10.2f}")

    best = max(rows, key=lambda r: r["mAP"])
    print(f"\nmelhor no split {args.split}: interior {best['interior_threshold']}, "
          f"foreground {best['fg_threshold']}, min_size {best['min_size']}  ->  mAP {best['mAP']:.4f}")
    # se o melhor cair na borda da grade, o otimo provavelmente ficou de fora dela
    for key, grid in [("interior_threshold", args.interior), ("fg_threshold", args.foreground),
                      ("min_size", args.min_size)]:
        if len(grid) > 1 and best[key] in (min(grid), max(grid)):
            print(f"  atencao: {key}={best[key]} caiu na borda da grade, vale estender")
    print("cola isso no bloco postprocess: do config e reavalia no teste")

    out = Path(args.out or Path(args.checkpoint).parent / "tune_watershed.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"split": args.split, "best": best, "grid": rows}, indent=2, default=float))
    print(f"salvou em {out}")


if __name__ == "__main__":
    main()
