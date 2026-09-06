#!/usr/bin/env python
"""Comando que treina.

    uv run python scripts/train.py --config configs/synthetic_baseline.yaml
    uv run python scripts/train.py --config configs/dsb2018_baseline.yaml
"""

from __future__ import annotations

import argparse

from pa1.engine.train import run
from pa1.utils import load_config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None, help="a Parte 3 pede 2 seeds por config")
    ap.add_argument("--device", default=None, help="cuda, mps ou cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["device"] = args.device

    run(cfg, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
