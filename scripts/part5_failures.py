#!/usr/bin/env python
"""Parte 5: galeria de falhas mais campo receptivo teorico.

Pega as 5 imagens de teste com pior AP, salva a figura de cada uma (imagem, ground
truth, predicao e os mapas intermediarios da trilha A) e junta os numeros que
sustentam o diagnostico de cada caso: quantos objetos, tamanho deles, quanto do erro
e fusao (uma previsao cobrindo varios gt) e quanto e fragmentacao (varias previsoes
cobrindo um gt).

Tambem calcula o campo receptivo teorico do encoder (slides 35 a 38) e compara com a
distribuicao de tamanho dos objetos do dataset, que e o item obrigatorio.

    uv run python scripts/part5_failures.py --config configs/dsb2018_boundary.yaml \
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
from pa1.metrics.instance import error_breakdown, evaluate_image
from pa1.receptive_field import ENCODERS, encoder_receptive_field, format_table, object_size_stats
from pa1.utils import get_device, load_checkpoint, load_config, seed_everything
from pa1.viz import panel, size_vs_receptive_field


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=5)
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

    out_dir = Path(args.out_dir or Path(args.checkpoint).parent / "part5_failures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. campo receptivo teorico contra tamanho dos objetos
    enc_name = cfg["model"].get("encoder_name", "resnet18")
    rf = encoder_receptive_field(enc_name)
    atrous_name = f"{enc_name}_atrous"
    rf_atrous = encoder_receptive_field(atrous_name) if atrous_name in ENCODERS else None
    print(format_table(rf))
    if rf_atrous:
        print(f"\ncom atrous no layer4: RF {rf_atrous['rf']} px, output stride {rf_atrous['output_stride']}")

    sizes = object_size_stats(dataset.raw(i)[1] for i in range(len(dataset)))
    print(f"\nobjetos do split {args.split}: n={sizes['n']}, mediana {sizes['median']:.1f} px, "
          f"p95 {sizes['p95']:.1f} px, max {sizes['max']:.1f} px")

    rf_lines = {"RF teorico": rf["rf"], "output stride": rf["output_stride"]}
    if rf_atrous:
        rf_lines["output stride c/ atrous"] = rf_atrous["output_stride"]
    size_vs_receptive_field(sizes["diameters"], rf_lines, path=out_dir / "receptive_field.png",
                            title=f"{enc_name}: campo receptivo contra tamanho dos nucleos")

    # 2. as n piores imagens
    per_image = []
    for i in range(len(dataset)):
        item = dataset[i]
        logits = model(item["image"].unsqueeze(0).to(device))
        pred = task.decode(logits)[0]
        maps = {k: v[0] for k, v in task.maps(logits).items()}
        _, gt = dataset.raw(i)
        h, w = gt.shape
        pred = pred[:h, :w]
        res = evaluate_image(pred, gt)
        per_image.append({"idx": i, "image_id": dataset.image_ids[i], "ap": res.ap,
                          "n_gt": res.n_gt, "n_pred": res.n_pred,
                          "breakdown": error_breakdown(pred, gt),
                          "sizes": object_size_stats([gt])})

    per_image.sort(key=lambda r: r["ap"])
    worst = per_image[: args.n]

    gallery = []
    for rank, rec in enumerate(worst, start=1):
        i = rec["idx"]
        item = dataset[i]
        logits = model(item["image"].unsqueeze(0).to(device))
        pred = task.decode(logits)[0]
        maps = {k: v[0] for k, v in task.maps(logits).items()}
        image, gt = dataset.raw(i)
        h, w = gt.shape
        b = rec["breakdown"]
        title = (f"{rank}. {rec['image_id'][:10]}  AP {rec['ap']:.3f}  "
                 f"gt {b['n_gt']} pred {b['n_pred']}  fundidas {b['merged']} fragmentadas {b['fragmented']}  "
                 f"diametro mediano {rec['sizes']['median']:.0f} px")
        path = out_dir / f"failure_{rank}.png"
        panel(image, gt, pred[:h, :w], {k: v[:h, :w] for k, v in maps.items()}, title=title, path=path)
        gallery.append({**{k: v for k, v in rec.items() if k != "sizes"},
                        "sizes": {k: v for k, v in rec["sizes"].items() if k != "diameters"},
                        "figure": str(path)})
        print(f"  {title}")

    payload = {
        "encoder": enc_name,
        "receptive_field": {k: v for k, v in rf.items() if k != "rows"},
        "receptive_field_rows": rf["rows"],
        "receptive_field_atrous": {k: v for k, v in rf_atrous.items() if k != "rows"} if rf_atrous else None,
        "object_sizes": {k: v for k, v in sizes.items() if k != "diameters"},
        "gallery": gallery,
        "dataset_breakdown": {
            "merged_total": int(sum(r["breakdown"]["merged"] for r in per_image)),
            "fragmented_total": int(sum(r["breakdown"]["fragmented"] for r in per_image)),
            "missed_total": int(sum(r["breakdown"]["missed"] for r in per_image)),
            "spurious_total": int(sum(r["breakdown"]["spurious"] for r in per_image)),
        },
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nresumo do split inteiro: {payload['dataset_breakdown']}")
    print(f"salvou a galeria e results.json em {out_dir}")


if __name__ == "__main__":
    main()
