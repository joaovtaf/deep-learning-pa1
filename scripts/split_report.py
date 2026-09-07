#!/usr/bin/env python
"""Descreve os clusters de modalidade e checa se o split ficou estratificado.

O enunciado exige split estratificado por modalidade e justificado. O DSB2018 nao
traz essa coluna, entao a gente infere com k-means sobre estatistica de cor. Esse
script e o que mostra que os clusters sao interpretaveis (nao sao grupos aleatorios)
e que as proporcoes se mantem em treino, validacao e teste.

    uv run python scripts/split_report.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pa1.data.dsb2018 import cached_labels
from pa1.data.splits import image_descriptor, modality_clusters, read_image, stratified_split

FEATURES = ["gray_mean", "gray_std", "sat_mean", "gray_median", "frac_escuro", "frac_claro", "R-G", "B-G"]


def describe(root, ids, sample=40):
    feats = np.stack([image_descriptor(read_image(root, i)) for i in ids[:sample]])
    labels = [cached_labels(root, i) for i in ids[:sample]]
    diameters = []
    for lab in labels:
        areas = np.bincount(lab.ravel())[1:]
        areas = areas[areas > 0]
        diameters += list(2.0 * np.sqrt(areas / np.pi))
    return {
        "features": {k: float(v) for k, v in zip(FEATURES, feats.mean(0))},
        "instancias_por_imagem": float(np.mean([lab.max() for lab in labels])),
        "diametro_mediano": float(np.median(diameters)) if diameters else 0.0,
    }


def guess_modality(f):
    """Rotulo legivel pro cluster, so pra apresentacao. E leitura nossa, nao vem do dataset."""
    if f["frac_escuro"] > 0.5 and f["sat_mean"] < 0.05:
        return "fluorescencia (fundo preto, cinza)"
    if f["sat_mean"] > 0.2:
        return "histologia corada (roxo/rosa)"
    if f["frac_claro"] > 0.3:
        return "brightfield (fundo claro)"
    return "outra"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/dsb2018/stage1_train")
    ap.add_argument("--n-clusters", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/split_report.json")
    args = ap.parse_args()

    ids, clusters = modality_clusters(args.root, args.n_clusters, args.seed)
    ids = np.array(ids)
    splits = stratified_split(args.root, n_clusters=args.n_clusters, seed=args.seed)
    where = {i: c for i, c in zip(ids, clusters)}

    print(f"{len(ids)} imagens, {args.n_clusters} clusters de modalidade\n")
    header = f"{'cluster':>8}{'n':>5}" + "".join(f"{n:>12}" for n in FEATURES) + \
             f"{'nuc/img':>10}{'diam med':>10}  leitura"
    print(header)

    rows = []
    for c in range(args.n_clusters):
        sel = ids[clusters == c]
        d = describe(args.root, list(sel))
        guess = guess_modality(d["features"])
        rows.append({"cluster": c, "n": int(len(sel)), **d, "leitura": guess})
        print(f"{c:>8}{len(sel):>5}" + "".join(f"{d['features'][k]:>12.3f}" for k in FEATURES) +
              f"{d['instancias_por_imagem']:>10.1f}{d['diametro_mediano']:>10.1f}  {guess}")

    print(f"\nsplit: " + ", ".join(f"{k} {len(v)}" for k, v in splits.items()))
    print(f"\n{'cluster':>8}{'treino':>9}{'val':>7}{'teste':>7}{'% treino':>10}{'% teste':>10}")
    dist = []
    for c in range(args.n_clusters):
        counts = {k: sum(1 for i in v if where[i] == c) for k, v in splits.items()}
        p_tr = counts["train"] / max(len(splits["train"]), 1)
        p_te = counts["test"] / max(len(splits["test"]), 1)
        dist.append({"cluster": c, **counts, "frac_train": p_tr, "frac_test": p_te})
        print(f"{c:>8}{counts['train']:>9}{counts['val']:>7}{counts['test']:>7}{p_tr:>10.3f}{p_te:>10.3f}")

    gap = max(abs(d["frac_train"] - d["frac_test"]) for d in dist)
    print(f"\nmaior diferenca de proporcao entre treino e teste: {gap:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"clusters": rows, "distribuicao": dist, "max_gap": gap,
                               "tamanhos": {k: len(v) for k, v in splits.items()}}, indent=2, default=float))
    print(f"salvou em {out}")


if __name__ == "__main__":
    main()
