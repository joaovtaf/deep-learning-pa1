#!/usr/bin/env python
"""Copia de runs/ pra results/ tudo que a apresentacao cita.

runs/ ta no .gitignore porque enche de checkpoint, mas o enunciado pede que toda
tabela e figura da apresentacao tenha lastro no repo. Entao os JSON de metrica e os
PNG viram results/, que vai versionado, e o APRESENTACAO.md aponta pra la.

    uv run python scripts/collect_results.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# de onde, pra onde
MAP = [
    ("runs/class_stats.json", "class_stats.json"),
    ("runs/split_report.json", "split_report.json"),
    ("runs/oracle_ceiling_synthetic_test.json", "oracle_ceiling_synthetic_test.json"),
    ("runs/oracle_ceiling_dsb2018_test.json", "oracle_ceiling_dsb2018_test.json"),

    ("runs/synthetic_baseline/history.json", "parte0/synthetic_baseline_history.json"),
    ("runs/synthetic_baseline/summary.json", "parte0/synthetic_baseline_summary.json"),
    ("runs/synthetic_baseline/eval_test/metrics.json", "parte0/synthetic_baseline_metrics.json"),
    ("runs/synthetic_baseline/eval_test/density.png", "parte0/synthetic_baseline_density.png"),
    ("runs/synthetic_boundary/history.json", "parte0/synthetic_boundary_history.json"),
    ("runs/synthetic_boundary/summary.json", "parte0/synthetic_boundary_summary.json"),
    ("runs/synthetic_boundary/eval_test/metrics.json", "parte0/synthetic_boundary_metrics.json"),

    ("runs/dsb2018_baseline/history.json", "parte1/history.json"),
    ("runs/dsb2018_baseline/summary.json", "parte1/summary.json"),
    ("runs/dsb2018_baseline/eval_test/metrics.json", "parte1/metrics.json"),
    ("runs/dsb2018_baseline/eval_test/density.png", "parte1/density.png"),
    ("runs/dsb2018_baseline/eval_test/panels", "parte1/panels"),
    ("runs/dsb2018_baseline/eval_test_hungarian/metrics.json", "parte1/metrics_hungarian.json"),

    ("runs/dsb2018_boundary/history.json", "parte2/history.json"),
    ("runs/dsb2018_boundary/summary.json", "parte2/summary.json"),
    ("runs/dsb2018_boundary/eval_test/metrics.json", "parte2/metrics.json"),
    ("runs/dsb2018_boundary/eval_test/density.png", "parte2/density.png"),
    ("runs/dsb2018_boundary/eval_test/panels", "parte2/panels"),
    ("runs/comparison", "parte2/comparacao"),

    ("runs/ablation_axis2/results.json", "parte3/eixo2_results.json"),
    ("runs/ablation_axis2/map.png", "parte3/eixo2_map.png"),
    ("runs/ablation_axis2/dice.png", "parte3/eixo2_dice.png"),
    ("runs/ablation_axis2/count_error.png", "parte3/eixo2_count_error.png"),
    ("runs/ablation_axis3/results.json", "parte3/eixo3_results.json"),
    ("runs/ablation_axis3/map.png", "parte3/eixo3_map.png"),
    ("runs/ablation_axis3/dice.png", "parte3/eixo3_dice.png"),
    ("runs/ablation_axis3/count_error.png", "parte3/eixo3_count_error.png"),

    ("runs/dsb2018_boundary/part4_mosaic", "parte4"),
    ("runs/dsb2018_boundary/part5_failures", "parte5"),
    ("runs/dsb2018_boundary/part6_stress", "parte6"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    runs, out = Path(args.runs), Path(args.out)
    copied, missing = [], []
    for src_rel, dst_rel in MAP:
        src = Path(src_rel.replace("runs/", f"{runs}/", 1))
        dst = out / dst_rel
        if not src.exists():
            missing.append(str(src))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        copied.append(str(dst))

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"{len(copied)} artefatos copiados pra {out} ({total/1e6:.1f} MB)")
    if missing:
        print(f"\n{len(missing)} nao existem ainda (roda a parte correspondente antes):")
        for m in missing:
            print("  " + m)


if __name__ == "__main__":
    main()
