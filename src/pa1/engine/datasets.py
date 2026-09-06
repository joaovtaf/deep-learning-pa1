"""Monta os datasets e os dataloaders a partir do bloco data: do config."""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

from ..data.dsb2018 import DSB2018Dataset
from ..data.splits import stratified_split
from ..data.synthetic import SyntheticEllipsesDataset


def build_datasets(cfg: dict) -> dict[str, Dataset]:
    cfg = dict(cfg)
    name = cfg.pop("name").lower()

    if name == "synthetic":
        counts = cfg.get("lengths", {"train": 512, "val": 64, "test": 128})
        seeds = {"train": 1, "val": 2, "test": 3}  # seeds diferentes deixam os splits disjuntos
        return {
            split: SyntheticEllipsesDataset(
                length=counts[split],
                size=cfg.get("size", 128),
                seed=seeds[split],
                n_range=tuple(cfg.get("n_range", (5, 20))),
                radius_range=tuple(cfg.get("radius_range", (5, 18))),
            )
            for split in ("train", "val", "test")
        }

    if name == "dsb2018":
        root = cfg.get("root", "data/dsb2018/stage1_train")
        splits = stratified_split(
            root=root,
            fractions=tuple(cfg.get("fractions", (0.7, 0.15, 0.15))),
            n_clusters=cfg.get("n_clusters", 4),
            seed=cfg.get("split_seed", 0),
        )
        return {
            split: DSB2018Dataset(splits[split], root=root, split=split, crop=cfg.get("crop", 256))
            for split in ("train", "val", "test")
        }

    raise KeyError(f"dataset '{name}' nao existe, tem 'synthetic' e 'dsb2018'")


def build_loaders(datasets: dict[str, Dataset], cfg: dict) -> dict[str, DataLoader]:
    # val e test com batch 1 porque cada imagem tem um tamanho e a metrica de
    # instancia precisa rodar na resolucao original
    num_workers = cfg.get("num_workers", 0)
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=cfg.get("batch_size", 8),
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            persistent_workers=num_workers > 0,
        ),
        "val": DataLoader(datasets["val"], batch_size=1, shuffle=False, num_workers=num_workers),
        "test": DataLoader(datasets["test"], batch_size=1, shuffle=False, num_workers=num_workers),
    }
