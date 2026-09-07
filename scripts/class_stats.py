#!/usr/bin/env python
"""Frequencia das 3 classes da trilha A e o alpha da CE balanceada que sai dela.

Roda antes de treinar a Parte 2. O numero que sai daqui e o que a gente colou no
alpha do config, e e o que quantifica o "desbalanceamento severo" que o eixo 2 da
Parte 3 ataca.

Tambem varre a espessura da fronteira, porque a espessura muda o desbalanceamento:
casca fina deixa a classe 2 minuscula, casca grossa come o interior dos nucleos
pequenos e eles param de virar marcador no watershed.

    uv run python scripts/class_stats.py --config configs/dsb2018_boundary.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from pa1.data.targets import alpha_from_frequencies, class_frequencies, three_class_mask
from pa1.engine.datasets import build_datasets
from pa1.utils import load_config

STRUCT = np.ones((3, 3), bool)


def marker_health(labels, thickness):
    """Quantos nucleos sobrevivem como marcador unico depois da erosao.

    lost   o interior sumiu inteiro, o nucleo nao vira marcador e vira FN
    split  o interior rachou em dois, o nucleo vira dois objetos e vira FP
    """
    three = three_class_mask(labels, thickness)
    interior = three == 1
    lost = split = total = 0
    for lab, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        total += 1
        m = (labels[sl] == lab) & interior[sl]
        if not m.any():
            lost += 1
            continue
        if ndi.label(m, structure=STRUCT)[1] > 1:
            split += 1
    return total, lost, split


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--thickness", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = build_datasets(cfg["data"])[args.split]
    n = min(args.limit, len(dataset))
    labels_list = [dataset.raw(i)[1] for i in range(n)]
    print(f"{n} imagens do split {args.split}, {sum(int(l.max()) for l in labels_list)} instancias")

    rows = []
    print(f"\n{'espessura':>10}{'fundo':>10}{'interior':>10}{'fronteira':>11}"
          f"{'alpha (1/sqrt f)':>28}{'perdidos':>10}{'rachados':>10}")
    for t in args.thickness:
        freq = class_frequencies(labels_list, t)
        alpha = alpha_from_frequencies(freq, "sqrt_inverse")
        total = lost = split = 0
        for lab in labels_list:
            a, b, c = marker_health(lab, t)
            total += a
            lost += b
            split += c
        rows.append({"thickness": t, "freq": [float(f) for f in freq], "alpha_sqrt_inverse": alpha,
                     "alpha_inverse": alpha_from_frequencies(freq, "inverse"),
                     "n_instances": total, "markers_lost": lost, "markers_split": split})
        print(f"{t:>10}{freq[0]:>10.3f}{freq[1]:>10.3f}{freq[2]:>11.3f}"
              f"{str([round(a,2) for a in alpha]):>28}{lost:>10}{split:>10}")

    out = Path(args.out or "runs/class_stats.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, default=float))
    print(f"\nsalvou em {out}")


if __name__ == "__main__":
    main()
