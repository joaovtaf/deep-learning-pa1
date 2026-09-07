#!/usr/bin/env python
"""Parte 4: inferencia em mosaico, o que quebra e a correcao.

Monta um mosaico grande com imagens do split de teste, roda inferencia em tiles com
sobreposicao do jeito do slide 83, e compara tres decodificacoes:

  full        sem tiling nenhum, imagem inteira de uma vez. E o teto.
  per_tile    decodifica dentro de cada tile e cola so a parte interna. E o jeito
              ingenuo, que quebra.
  blend       media dos logits na sobreposicao e decodifica uma vez so no fim.
  fuse        decodifica por tile e costura por IoU na faixa de sobreposicao.

Mede o mAP das quatro e salva as figuras, incluindo o zoom numa emenda mostrando o
que acontece com um objeto que cai em cima dela.

    uv run python scripts/part4_mosaic.py --config configs/dsb2018_boundary.yaml \
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
from pa1.metrics.instance import evaluate_image, instance_iou_matrix
from pa1.tiling import (
    blend_logits,
    build_mosaic,
    fuse_tile_instances,
    paste_per_tile_labels,
    predict_tiles,
    tile_positions,
)
from pa1.utils import as_rgb_uint8, get_device, load_checkpoint, load_config, seed_everything
from pa1.viz import mosaic_figure, seam_zoom


@torch.no_grad()
def predict_full(model, task, image, device):
    """Imagem inteira, com padding pra fechar multiplo de 32."""
    h, w = image.shape[:2]
    ph, pw = (-h) % 32, (-w) % 32
    padded = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    x = torch.from_numpy(padded.astype(np.float32).transpose(2, 0, 1)[None] / 255.0).to(device)
    return task.decode(model(x))[0][:h, :w]


def seam_objects(gt_labels, seam_x, band=6):
    """Instancias do ground truth que cruzam uma emenda vertical."""
    strip = gt_labels[:, max(0, seam_x - band) : seam_x + band]
    ids = np.unique(strip)
    ids = ids[ids > 0]
    crossing = []
    for i in ids:
        cols = np.nonzero((gt_labels == i).any(axis=0))[0]
        if cols.min() < seam_x < cols.max():
            crossing.append(int(i))
    return crossing


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--grid", type=int, nargs=2, default=[3, 3])
    ap.add_argument("--cell", type=int, default=256)
    ap.add_argument("--iou-fuse", type=float, default=0.25)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 0)))
    device = get_device(args.device or cfg.get("device", "auto"))

    datasets = build_datasets(cfg["data"])
    model, task = build_from_config(cfg, device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    test = datasets["test"]
    # escolhe as imagens mais densas, que e onde a emenda tem mais chance de cortar
    # um nucleo no meio
    counts = [(int(test.raw(i)[1].max()), i) for i in range(len(test))]
    counts.sort(reverse=True)
    indices = [i for _, i in counts[: args.grid[0] * args.grid[1]]]

    image, gt = build_mosaic(test, indices, tuple(args.grid), tile_size=args.cell)
    print(f"mosaico {image.shape[0]}x{image.shape[1]} com {int(gt.max())} instancias no ground truth")

    logits, coords = predict_tiles(model, image, device, tile=args.tile, overlap=args.overlap)
    per_tile = task.decode(logits)
    shape = image.shape[:2]

    variants = {}
    variants["full"] = predict_full(model, task, image, device)
    variants["per_tile"] = paste_per_tile_labels(per_tile, coords, shape, args.tile, args.overlap)
    blended = blend_logits(logits, coords, shape, args.tile, args.overlap)
    variants["blend"] = task.decode(blended.unsqueeze(0))[0]
    variants["fuse"] = fuse_tile_instances(per_tile, coords, shape, args.tile, iou_threshold=args.iou_fuse)

    results = {}
    for name, lab in variants.items():
        r = evaluate_image(lab, gt)
        results[name] = {"mAP": r.ap, "n_pred": r.n_pred, "n_gt": r.n_gt, "count_error": r.count_error,
                         "per_threshold": {str(k): v for k, v in r.per_threshold.items()}}
        print(f"  {name:<10} mAP {r.ap:.4f}  objetos previstos {r.n_pred} (gt {r.n_gt})")

    # o que acontece com um objeto na emenda: pega a primeira emenda vertical
    xs = tile_positions(shape[1], args.tile, args.overlap)
    seam_x = xs[1] + args.overlap // 2 if len(xs) > 1 else shape[1] // 2
    crossing = seam_objects(gt, seam_x)
    print(f"  emenda em x={seam_x}, {len(crossing)} instancias do gt cruzam ela")

    seam_detail = {}
    for name in ("per_tile", "fuse"):
        lab = variants[name]
        iou = instance_iou_matrix(lab, gt)
        # so as instancias do gt que cruzam a emenda
        best = [float(iou[:, i - 1].max()) if iou.size else 0.0 for i in crossing]
        seam_detail[name] = {"n_crossing": len(crossing), "mean_best_iou": float(np.mean(best)) if best else 0.0,
                             "recovered_at_050": int(sum(b >= 0.5 for b in best))}
        print(f"  {name:<10} nas {len(crossing)} instancias da emenda: IoU medio do melhor par "
              f"{seam_detail[name]['mean_best_iou']:.3f}, recuperadas em IoU 0.5: "
              f"{seam_detail[name]['recovered_at_050']}")

    out_dir = Path(args.out_dir or Path(args.checkpoint).parent / "part4_mosaic")
    out_dir.mkdir(parents=True, exist_ok=True)
    mosaic_figure(image, gt, variants, path=out_dir / "mosaic.png",
                  title=f"Parte 4: tile {args.tile}, sobreposicao {args.overlap}")

    ys = tile_positions(shape[0], args.tile, args.overlap)
    y0 = ys[0]
    window = (y0, min(y0 + 220, shape[0]), max(0, seam_x - 110), min(seam_x + 110, shape[1]))
    seam_zoom(image, {k: variants[k] for k in ("per_tile", "fuse", "blend")}, window,
              path=out_dir / "seam.png", title=f"objeto na emenda entre dois tiles (x={seam_x})")

    payload = {"tile": args.tile, "overlap": args.overlap, "grid": args.grid, "cell": args.cell,
               "n_tiles": len(coords), "mosaic_shape": list(shape), "results": results,
               "seam": {"x": seam_x, "crossing_ids": crossing, "detail": seam_detail}}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nsalvou mosaic.png, seam.png e results.json em {out_dir}")


if __name__ == "__main__":
    main()
