#!/usr/bin/env python
"""Parte 3, eixos 2 e 3. Cada configuracao roda com 2 seeds e sai media +- desvio.

Escolhemos os eixos 2 e 3. O eixo 1 (como recuperar resolucao) ficou de fora porque
o nosso decoder e o mesmo em todas as configuracoes e trocar o mecanismo de upsample
mudaria a arquitetura junto com a variavel que a gente quer isolar.

Eixo 2, a funcao de perda. CE -> CE balanceada (alpha) -> focal (gamma) -> focal
balanceada, slides 73 a 79, com gamma em {0, 1, 2, 5} como o enunciado pede. O
desbalanceamento aqui e real: a classe fronteira e uma fracao pequena da imagem e o
fundo domina a soma da CE.

Eixo 3, contexto global. Sem contexto, image pooling do ParseNet e pyramid pooling do
PSPNet, slides 51 a 55, plugados no mesmo lugar (topo do encoder). A pergunta do
enunciado e se contexto global ajuda a SEPARAR instancias ou so a classifica-las
melhor, entao a tabela reporta as duas coisas lado a lado: Dice (classificar) e mAP
com erro de contagem (separar).

    uv run python scripts/ablations.py --config configs/dsb2018_boundary.yaml --axis 2
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from pa1.engine.train import run
from pa1.utils import load_config
from pa1.viz import ablation_bars

# alpha vem da frequencia real das 3 classes no treino, ver scripts/class_stats.py
ALPHA = [0.45, 0.80, 1.75]

AXIS2 = [
    ("ce",            "CE (gamma=0)",             {"name": "ce"}),
    ("ce_bal",        "CE balanceada",            {"name": "ce", "alpha": ALPHA}),
    ("focal_g1",      "focal gamma=1",            {"name": "focal", "gamma": 1.0}),
    ("focal_g2",      "focal gamma=2",            {"name": "focal", "gamma": 2.0}),
    ("focal_g5",      "focal gamma=5",            {"name": "focal", "gamma": 5.0}),
    ("focal_g2_bal",  "focal balanceada gamma=2", {"name": "focal", "gamma": 2.0, "alpha": ALPHA}),
]

AXIS3 = [
    ("none",            "sem contexto"),
    ("image_pooling",   "image pooling (ParseNet)"),
    ("pyramid_pooling", "pyramid pooling (PSPNet)"),
]


def configs_for_axis(base: dict, axis: int):
    out = []
    if axis == 2:
        for key, label, loss_cfg in AXIS2:
            cfg = copy.deepcopy(base)
            cfg["task"]["loss_cfg"] = loss_cfg
            out.append((key, label, cfg))
    elif axis == 3:
        for key, label in AXIS3:
            cfg = copy.deepcopy(base)
            cfg["model"]["context"] = key
            out.append((key, label, cfg))
    else:
        raise ValueError("so implementamos os eixos 2 e 3")
    return out


def aggregate(runs):
    """media +- desvio sobre as seeds, no formato que a Parte 3 pede."""
    keys = ["mAP", "AP50", "iou", "dice", "mean_count_error"]
    out = {}
    for k in keys:
        vals = np.array([r["final_val"][k] for r in runs], dtype=float)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=0))
        out[f"{k}_values"] = [float(v) for v in vals]
    out["minutes"] = float(np.sum([r["minutes"] for r in runs]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--axis", type=int, required=True, choices=[2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=None, help="orcamento reduzido pras ablacoes")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    base = load_config(args.config)
    if args.epochs:
        base["epochs"] = args.epochs
    if args.device:
        base["device"] = args.device

    out_dir = Path(args.out_dir or f"runs/ablation_axis{args.axis}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    started = time.time()
    for key, label, cfg in configs_for_axis(base, args.axis):
        per_seed = []
        for seed in args.seeds:
            cfg_seed = copy.deepcopy(cfg)
            cfg_seed["seed"] = seed
            cfg_seed["name"] = f"axis{args.axis}_{key}_s{seed}"
            print(f"\n########## eixo {args.axis} | {label} | seed {seed}")
            per_seed.append(run(cfg_seed, output_dir=out_dir / f"{key}_s{seed}"))

        row = {"key": key, "label": label, **aggregate(per_seed)}
        rows.append(row)
        print(f"\n>>> {label}: mAP {row['mAP_mean']:.4f} +- {row['mAP_std']:.4f}  "
              f"Dice {row['dice_mean']:.4f} +- {row['dice_std']:.4f}")
        (out_dir / "results.json").write_text(json.dumps(rows, indent=2, default=float))

    ablation_bars(rows, "mAP", out_dir / "map.png",
                  title=f"Parte 3, eixo {args.axis}: mAP de validacao (media +- desvio, 2 seeds)")
    ablation_bars(rows, "dice", out_dir / "dice.png",
                  title=f"Parte 3, eixo {args.axis}: Dice de validacao (media +- desvio, 2 seeds)")
    ablation_bars(rows, "mean_count_error", out_dir / "count_error.png",
                  title=f"Parte 3, eixo {args.axis}: erro de contagem (media +- desvio, 2 seeds)")

    print(f"\n=== eixo {args.axis}, {len(rows)} configuracoes, {(time.time()-started)/60:.1f} min")
    print(f"{'config':<28}{'mAP':>18}{'Dice':>18}{'|err contagem|':>18}")
    for r in rows:
        print(f"{r['label']:<28}{r['mAP_mean']:>10.4f} +-{r['mAP_std']:<5.3f}"
              f"{r['dice_mean']:>10.4f} +-{r['dice_std']:<5.3f}"
              f"{r['mean_count_error_mean']:>10.2f} +-{r['mean_count_error_std']:<5.2f}")


if __name__ == "__main__":
    main()
