"""Loop de avaliacao: IoU/Dice semanticos e mAP/erro de contagem de instancia.

Roda com batch 1 na resolucao cheia, senao o matching veria uma geometria
diferente da que o modelo vai ver na inferencia.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..metrics.instance import DEFAULT_THRESHOLDS, InstanceMeter
from ..metrics.semantic import SemanticMeter
from ..postprocess.naive import labels_from_probability
from ..utils import tqdm_disabled


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    min_size: int = 10,
    matching: str = "greedy",
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    progress: bool = True,
) -> dict:
    model.eval()
    sem = SemanticMeter(threshold=threshold)
    inst = InstanceMeter(thresholds=thresholds, method=matching)
    image_ids: list[str] = []

    iterator = tqdm(loader, desc="eval", leave=False, disable=tqdm_disabled()) if progress else loader
    for batch in iterator:
        images = batch["image"].to(device)
        probs = torch.sigmoid(model(images)).cpu().numpy()[:, 0]
        gt_labels = batch["labels"].cpu().numpy()
        gt_mask = batch["mask"].cpu().numpy()[:, 0]

        for b in range(probs.shape[0]):
            sem.update(probs[b], gt_mask[b])
            pred_labels = labels_from_probability(probs[b], threshold=threshold, min_size=min_size)
            inst.update(pred_labels, gt_labels[b])
        image_ids += list(batch.get("image_id", [""] * probs.shape[0]))

    out = {**sem.compute(), **inst.compute(), "matching": matching}
    out["_meter"] = inst
    out["_image_ids"] = image_ids
    return out


def format_report(metrics: dict, title: str = "") -> str:
    lines = [f"== {title}" if title else "=="]
    lines.append(f"  imagens       : {metrics['n']}")
    lines.append(f"  IoU (semantic): {metrics['iou']:.4f}")
    lines.append(f"  Dice          : {metrics['dice']:.4f}")
    lines.append(f"  mAP  @.50:.95 : {metrics['mAP']:.4f}   (matching: {metrics['matching']})")
    lines.append(f"  AP   @.50     : {metrics['AP50']:.4f}")
    lines.append(f"  |erro contagem|: {metrics['mean_count_error']:.2f} por imagem")
    per_t = metrics.get("per_threshold", {})
    if per_t:
        cells = "  ".join(f"{t:.2f}:{v:.3f}" for t, v in sorted(per_t.items()))
        lines.append(f"  por limiar    : {cells}")
    return "\n".join(lines)


def save_metrics(metrics: dict, path: str | Path) -> None:
    """Salva as metricas junto com a tabela por imagem que o grafico de densidade usa."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meter: InstanceMeter = metrics["_meter"]
    payload = {k: v for k, v in metrics.items() if not k.startswith("_")}
    ids = metrics["_image_ids"] or [""] * len(meter.results)
    payload["per_image"] = [
        {"image_id": iid, "ap": r.ap, "n_gt": r.n_gt, "n_pred": r.n_pred, "count_error": r.count_error}
        for iid, r in zip(ids, meter.results)
    ]
    path.write_text(json.dumps(payload, indent=2, default=float))


def density_arrays(metrics: dict):
    """(densidade, ap, erro de contagem) pro grafico do item 5 da Parte 1."""
    table = metrics["_meter"].density_table()
    return table[:, 0], table[:, 1], table[:, 2]
