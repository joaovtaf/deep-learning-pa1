#!/usr/bin/env python
"""Quantifica o item 5 da Parte 1: mAP e erro de contagem contra densidade de objetos.

O grafico ja sai do scripts/evaluate.py, mas "a tendencia tem que ficar visivel" pede um
numero pra falar na apresentacao, nao so uma nuvem de pontos. Aqui saem a inclinacao da
reta, a correlacao e a media por faixa de densidade, pros modelos que a gente quiser
comparar lado a lado.

    uv run python scripts/density_trend.py \
        --metrics "Parte 1" results/parte1/metrics.json \
                  "Parte 2" results/parte2/metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BINS = [(1, 10), (11, 20), (21, 40), (41, 80), (81, 10**6)]


def load(path):
    d = json.loads(Path(path).read_text())
    rows = d["per_image"]
    return (np.array([r["n_gt"] for r in rows], dtype=float),
            np.array([r["ap"] for r in rows], dtype=float),
            np.array([r["count_error"] for r in rows], dtype=float))


def _corr(x, y):
    # se um dos dois for constante a correlacao nao existe, nao adianta deixar dar nan
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def describe(name, density, ap, cerr):
    slope_ap = np.polyfit(density, ap, 1)[0]
    slope_ce = np.polyfit(density, cerr, 1)[0]
    r_ap = _corr(density, ap)
    r_ce = _corr(density, cerr)
    print(f"\n{name}: {len(density)} imagens, densidade de {int(density.min())} a "
          f"{int(density.max())}, mediana {int(np.median(density))}")
    print(f"  AP contra densidade   : inclinacao {slope_ap:+.5f} por objeto, correlacao {r_ap:+.3f}")
    print(f"  |erro| contra densidade: inclinacao {slope_ce:+.5f} por objeto, correlacao {r_ce:+.3f}")

    print(f"  {'faixa':<12}{'n':>5}{'AP medio':>11}{'|erro| medio':>15}")
    per_bin = []
    for lo, hi in BINS:
        m = (density >= lo) & (density <= hi)
        if not m.any():
            continue
        label = f"{lo}-{hi}" if hi < 10**5 else f"{lo}+"
        print(f"  {label:<12}{int(m.sum()):>5}{ap[m].mean():>11.4f}{cerr[m].mean():>15.2f}")
        per_bin.append({"faixa": label, "n": int(m.sum()), "ap": float(ap[m].mean()),
                        "count_error": float(cerr[m].mean())})
    return {"nome": name, "slope_ap": float(slope_ap), "slope_count_error": float(slope_ce),
            "corr_ap": float(r_ap), "corr_count_error": float(r_ce), "faixas": per_bin}


def main():
    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.add_argument("--metrics", nargs="+", required=True,
                     help="pares nome caminho, ex: 'Parte 1' results/parte1/metrics.json")
    ap_.add_argument("--out", default="runs/density_trend.json")
    args = ap_.parse_args()

    pairs = list(zip(args.metrics[::2], args.metrics[1::2]))
    out = [describe(name, *load(path)) for name, path in pairs]

    if len(out) == 2:
        a, b = out
        print(f"\n{a['nome']} perde {abs(a['slope_ap'])*100:.2f} pontos de AP a cada 100 objetos, "
              f"{b['nome']} perde {abs(b['slope_ap'])*100:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nsalvou em {args.out}")


if __name__ == "__main__":
    main()
