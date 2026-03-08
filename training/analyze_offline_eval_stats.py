#!/usr/bin/env python3
"""Aggregate OMNeT++ .sca scalar outputs with mean/std/95% CI.

Usage:
  python3 training/analyze_offline_eval_stats.py \
    --input results/offline_eval \
    --output results/offline_eval/stats
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate OMNeT++ scalar stats")
    p.add_argument("--input", required=True, help="Root containing .sca files")
    p.add_argument("--output", required=True, help="Output directory for reports")
    p.add_argument(
        "--metrics",
        default="forwardedPackets,successfulForwards,failedForwards,queueDrops,uavCollectedPackets,uavUploadedPackets",
        help="Comma-separated scalar names to extract",
    )
    return p.parse_args()


def mean_std_ci(values: List[float]) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, 0.0
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(var)
    ci95 = 1.96 * std / math.sqrt(n)
    return mean, std, ci95


def main() -> None:
    args = parse_args()
    in_root = Path(args.input)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    wanted = {m.strip() for m in args.metrics.split(",") if m.strip()}
    sca_files = sorted(in_root.rglob("*.sca"))
    if not sca_files:
        raise SystemExit(f"No .sca files found under {in_root}")

    # key: (config_name, metric) -> values across runs/modules
    bucket: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    for sca in sca_files:
        config_name = "unknown"
        with sca.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith("attr configname "):
                    config_name = line.split(" ", 2)[-1]
                    continue
                if not line.startswith("scalar "):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                metric = parts[2]
                if metric not in wanted:
                    continue
                try:
                    value = float(parts[3])
                except ValueError:
                    continue
                bucket[(config_name, metric)].append(value)

    report = {"input_root": str(in_root), "rows": []}
    csv_lines = ["config,metric,n,mean,std,ci95"]

    for (config, metric), values in sorted(bucket.items()):
        mean, std, ci95 = mean_std_ci(values)
        row = {
            "config": config,
            "metric": metric,
            "n": len(values),
            "mean": mean,
            "std": std,
            "ci95": ci95,
        }
        report["rows"].append(row)
        csv_lines.append(f"{config},{metric},{len(values)},{mean},{std},{ci95}")

    (out_root / "eval_stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_root / "eval_stats.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    print(f"Wrote {out_root / 'eval_stats.json'}")
    print(f"Wrote {out_root / 'eval_stats.csv'}")


if __name__ == "__main__":
    main()
