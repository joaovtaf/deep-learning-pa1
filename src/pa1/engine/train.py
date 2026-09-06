"""Loop de treino do baseline binario da Parte 1.

O modelo salvo e o de melhor mAP de validacao, nao o de menor loss. Faz
diferenca, porque da pra ter Dice otimo e mAP horrivel, que e justamente o que a
Parte 1 quer mostrar.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..losses import build_loss
from ..models.unet import build_model
from ..utils import AverageMeter, get_device, save_checkpoint, seed_everything, tqdm_disabled
from .datasets import build_datasets, build_loaders
from .evaluate import evaluate, format_report


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> float:
    model.train()
    meter = AverageMeter()
    bar = tqdm(loader, desc=f"epoch {epoch}/{epochs}", leave=False, disable=tqdm_disabled())
    for batch in bar:
        images = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), target)
        loss.backward()
        optimizer.step()

        meter.update(loss.item(), images.size(0))
        bar.set_postfix(loss=f"{meter.avg:.4f}")
    return meter.avg


def run(cfg: dict, output_dir: str | Path | None = None) -> dict:
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    device = get_device(cfg.get("device", "auto"))

    out_dir = Path(output_dir or cfg.get("output_dir", "runs/default"))
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_datasets(cfg["data"])
    loaders = build_loaders(datasets, cfg.get("loader", {}))

    model = build_model(cfg["model"]).to(device)
    criterion = build_loss(cfg["loss"])
    opt_cfg = cfg.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 3e-4)),
        weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(cfg.get("epochs", 10))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    pp = cfg.get("postprocess", {})
    eval_kwargs = dict(
        threshold=float(pp.get("threshold", 0.5)),
        min_size=int(pp.get("min_size", 10)),
        matching=cfg.get("matching", "greedy"),
    )

    print(
        f"device: {device} | seed: {seed} | train/val/test: "
        f"{len(datasets['train'])}/{len(datasets['val'])}/{len(datasets['test'])}"
    )

    history = []
    best_map = -1.0
    start = time.time()

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device, epoch, epochs)
        scheduler.step()
        metrics = evaluate(model, loaders["val"], device, progress=False, **eval_kwargs)
        elapsed = time.time() - start
        print(
            f"epoch {epoch:3d}/{epochs} | loss {loss:.4f} | val IoU {metrics['iou']:.4f} "
            f"Dice {metrics['dice']:.4f} mAP {metrics['mAP']:.4f} | {elapsed / 60:.1f} min"
        )
        history.append(
            {"epoch": epoch, "loss": loss, "iou": metrics["iou"], "dice": metrics["dice"], "mAP": metrics["mAP"]}
        )
        if metrics["mAP"] > best_map:
            best_map = metrics["mAP"]
            save_checkpoint(
                out_dir / "best.pt", model, meta={"config": cfg, "epoch": epoch, "val_mAP": best_map, "seed": seed}
            )

    save_checkpoint(out_dir / "last.pt", model, meta={"config": cfg, "epoch": epochs, "seed": seed})
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    total_min = (time.time() - start) / 60
    print(f"\ntreino terminou em {total_min:.2f} min | melhor mAP de val {best_map:.4f}")
    print(format_report(evaluate(model, loaders["val"], device, **eval_kwargs), "modelo final, split de validacao"))
    return {"best_val_mAP": best_map, "minutes": total_min, "history": history, "output_dir": str(out_dir)}
