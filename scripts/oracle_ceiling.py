#!/usr/bin/env python
"""Teto de oraculo: quanto cada decodificacao entregaria se a rede acertasse tudo.

Nao treina nada e nao carrega checkpoint. Alimenta cada pos-processamento com o alvo
perfeito derivado do ground truth e mede o mAP resultante. Responde a pergunta que
separa as Partes 1 e 2: o gargalo esta na rede ou na representacao?

Se o teto do componentes conexos ja for baixo, nenhuma rede melhor resolve, porque a
informacao que separa dois nucleos encostados nao existe numa mascara binaria. Rodar
isso antes de treinar economizou bastante tempo nosso.

    uv run python scripts/oracle_ceiling.py --config configs/dsb2018_boundary.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pa1.data.targets import distance_map, semantic_mask, three_class_mask
from pa1.engine.datasets import build_datasets
from pa1.metrics.instance import evaluate_image
from pa1.postprocess.naive import labels_from_probability
from pa1.postprocess.watershed import labels_from_boundary_head, labels_from_distance_only
from pa1.utils import load_config


def oracle_probs(labels, thickness):
    """As tres probabilidades de classe que uma rede perfeita cuspiria."""
    three = three_class_mask(labels, thickness)
    return np.stack([three == 0, three == 1, three == 2]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--thickness", type=int, default=None)
    ap.add_argument("--min-size", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = build_datasets(cfg["data"])[args.split]
    thickness = args.thickness or cfg["data"].get("boundary_thickness", 2)
    min_size = args.min_size or cfg.get("postprocess", {}).get("min_size", 20)

    n = min(args.limit or len(dataset), len(dataset))
    rows = []
    for i in range(n):
        _, labels = dataset.raw(i)
        probs = oracle_probs(labels, thickness)
        dist = distance_map(labels)
        rows.append({
            "image_id": dataset.image_ids[i],
            "n_gt": int(labels.max()),
            "cc": evaluate_image(labels_from_probability(semantic_mask(labels), min_size=min_size), labels).ap,
            "watershed": evaluate_image(labels_from_boundary_head(probs, dist, min_size=min_size), labels).ap,
            "distancia": evaluate_image(labels_from_distance_only(dist, 0.05, 0.6, min_size=min_size), labels).ap,
        })

    counts = np.array([r["n_gt"] for r in rows])
    print(f"{n} imagens do split {args.split}, {counts.sum()} instancias, "
          f"mediana de {int(np.median(counts))} por imagem, espessura {thickness}\n")

    print(f"{'decodificacao':<34}{'entrada perfeita':<26}{'mAP':>8}")
    for key, desc in [("cc", "mascara semantica"), ("watershed", "fronteira e distancia"),
                      ("distancia", "so o mapa de distancia")]:
        vals = np.array([r[key] for r in rows])
        name = {"cc": "limiar + componentes conexos", "watershed": "watershed com marcadores",
                "distancia": "watershed so da distancia"}[key]
        print(f"{name:<34}{desc:<26}{vals.mean():>8.4f}")

    # o mesmo corte so nas imagens mais densas, que e onde a diferenca importa
    dense = np.argsort(counts)[-max(len(rows) // 3, 1):]
    print(f"\nso no terco mais denso (>= {counts[dense].min()} instancias por imagem):")
    for key in ("cc", "watershed", "distancia"):
        vals = np.array([r[key] for r in rows])[dense]
        print(f"  {key:<12}{vals.mean():>8.4f}")

    payload = {"split": args.split, "n_images": n, "thickness": thickness, "min_size": min_size,
               "media": {k: float(np.mean([r[k] for r in rows])) for k in ("cc", "watershed", "distancia")},
               "terco_mais_denso": {k: float(np.mean(np.array([r[k] for r in rows])[dense]))
                                    for k in ("cc", "watershed", "distancia")},
               "por_imagem": rows}
    out = Path(args.out or f"runs/oracle_ceiling_{cfg['data']['name']}_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nsalvou em {out}")


if __name__ == "__main__":
    main()
