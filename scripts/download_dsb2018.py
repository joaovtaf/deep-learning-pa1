#!/usr/bin/env python
"""Baixa e extrai o DSB2018 (BBBC038v1). Nao precisa de conta.

    uv run python scripts/download_dsb2018.py
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

URLS = {
    "stage1_train": "https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip",
    "stage1_test": "https://data.broadinstitute.org/bbbc/BBBC038/stage1_test.zip",
}


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"{dest} ja baixado, pulando")
        return dest
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
        tmp.rename(dest)
    return dest


def extract(zip_path: Path, out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"{out_dir} ja extraido, pulando")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in tqdm(zf.namelist(), desc=f"extraindo {out_dir.name}"):
            zf.extract(member, out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/dsb2018")
    ap.add_argument("--splits", nargs="+", default=["stage1_train"], choices=sorted(URLS))
    args = ap.parse_args()

    root = Path(args.root)
    for split in args.splits:
        zip_path = download(URLS[split], root / f"{split}.zip")
        extract(zip_path, root / split)

    train_dir = root / "stage1_train"
    if train_dir.exists():
        n = sum(1 for p in train_dir.iterdir() if (p / "images").is_dir())
        print(f"\n{n} imagens de treino em {train_dir}")


if __name__ == "__main__":
    main()
