"""Loop de treino, serve pras duas tarefas.

O modelo salvo e o de melhor mAP de validacao, nao o de menor loss. Faz diferenca,
porque da pra ter Dice otimo e mAP horrivel, que e justamente o que a Parte 1 quer
mostrar.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.unet import build_model
from ..utils import AverageMeter, get_device, save_checkpoint, seed_everything, tqdm_disabled
from .datasets import build_datasets, build_loaders
from .evaluate import evaluate, format_report
from .tasks import build_task


def train_one_epoch(model, loader, task, optimizer, device, epoch, epochs) -> float:
    model.train()
    meter = AverageMeter()
    bar = tqdm(loader, desc=f"epoch {epoch}/{epochs}", leave=False, disable=tqdm_disabled())
    for batch in bar:
        images = batch["image"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = task.loss(model(images), batch, device)
        loss.backward()
        optimizer.step()
        meter.update(loss.item(), images.size(0))
        bar.set_postfix(loss=f"{meter.avg:.4f}")
    return meter.avg


def build_from_config(cfg: dict, device):
    """Modelo e task saem juntos porque o numero de canais da head vem da task."""
    task = build_task({**cfg["task"], "postprocess": cfg.get("postprocess", {})})
    model_cfg = dict(cfg["model"])
    model_cfg.setdefault("out_channels", task.out_channels)
    model = build_model(model_cfg).to(device)
    return model, task


def run(cfg: dict, output_dir: str | Path | None = None) -> dict:
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    device = get_device(cfg.get("device", "auto"))

    out_dir = Path(output_dir or cfg.get("output_dir", "runs/default"))
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_datasets(cfg["data"])
    loaders = build_loaders(datasets, cfg.get("loader", {}))
    model, task = build_from_config(cfg, device)

    opt_cfg = cfg.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 3e-4)),
        weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(cfg.get("epochs", 10))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    matching = cfg.get("matching", "greedy")
    eval_every = int(cfg.get("eval_every", 1))

    print(
        f"device: {device} | seed: {seed} | task: {task.name} | train/val/test: "
        f"{len(datasets['train'])}/{len(datasets['val'])}/{len(datasets['test'])}"
    )

    history = []
    best_map = -1.0
    start = time.time()

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, loaders["train"], task, optimizer, device, epoch, epochs)
        scheduler.step()

        if epoch % eval_every and epoch != epochs:
            print(f"epoch {epoch:3d}/{epochs} | loss {loss:.4f} | {(time.time() - start) / 60:.1f} min")
            history.append({"epoch": epoch, "loss": loss})
            continue

        m = evaluate(model, loaders["val"], device, task, matching=matching, progress=False)
        elapsed = time.time() - start
        print(
            f"epoch {epoch:3d}/{epochs} | loss {loss:.4f} | val IoU {m['iou']:.4f} "
            f"Dice {m['dice']:.4f} mAP {m['mAP']:.4f} | {elapsed / 60:.1f} min"
        )
        history.append(
            {"epoch": epoch, "loss": loss, "iou": m["iou"], "dice": m["dice"], "mAP": m["mAP"]}
        )
        if m["mAP"] > best_map:
            best_map = m["mAP"]
            save_checkpoint(
                out_dir / "best.pt", model, meta={"config": cfg, "epoch": epoch, "val_mAP": best_map, "seed": seed}
            )

    save_checkpoint(out_dir / "last.pt", model, meta={"config": cfg, "epoch": epochs, "seed": seed})
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    total_min = (time.time() - start) / 60
    final = evaluate(model, loaders["val"], device, task, matching=matching, progress=False)
    print(f"\ntreino terminou em {total_min:.2f} min | melhor mAP de val {best_map:.4f}")
    print(format_report(final, "modelo final, split de validacao"))

    summary = {
        "name": cfg.get("name", out_dir.name),
        "seed": seed,
        "best_val_mAP": best_map,
        "minutes": total_min,
        "output_dir": str(out_dir),
        "final_val": {k: v for k, v in final.items() if not k.startswith("_")},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary
