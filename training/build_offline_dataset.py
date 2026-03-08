#!/usr/bin/env python3
"""Build, validate, and split offline transitions from OMNeT++ outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "run_id",
    "seed",
    "config_name",
    "mobility_tag",
    "policy_tag",
    "policy_mode",
    "epsilon_or_temperature",
    "exploration_tier",
    "node_id",
    "packet_id",
    "state",
    "candidates",
    "action_index",
    "action_features",
    "reward",
    "next_state_context",
    "next_candidates",
    "done",
    "schema_version",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build deterministic offline RL dataset manifests")
    p.add_argument("--input", required=True, help="Root folder containing run-* subfolders")
    p.add_argument("--output", required=True, help="Output folder for split manifests")
    p.add_argument("--train-max-seed", type=int, default=59)
    p.add_argument("--val-max-seed", type=int, default=79)
    p.add_argument("--min-policy-samples", type=int, default=100)
    return p.parse_args()


def find_transition_files(root: Path) -> list[Path]:
    sharded = sorted(root.glob("run-*/transitions-node*.jsonl"))
    if sharded:
        return sharded
    # Fallback for legacy single-file naming.
    legacy = sorted(root.glob("run-*/transitions.jsonl"))
    if legacy:
        return legacy
    return sorted(root.glob("run-*/transitions*.jsonl"))


def split_name(seed: int, train_max: int, val_max: int) -> str:
    if seed <= train_max:
        return "train"
    if seed <= val_max:
        return "val"
    return "test"


def main() -> None:
    args = parse_args()
    in_root = Path(args.input)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    files = find_transition_files(in_root)
    if not files:
        raise SystemExit(f"No transitions found under {in_root}")

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    policy_counts: Counter[str] = Counter()
    exploration_counts: Counter[str] = Counter()
    seed_to_split: dict[int, str] = {}
    errors: list[str] = []

    total = 0
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"{path}:{ln} invalid json: {e}")
                    continue

                if rec.get("event") != "transition":
                    continue

                # Backward-compatible aliases for earlier records.
                if "node_id" not in rec and "node" in rec:
                    rec["node_id"] = rec["node"]
                if "packet_id" not in rec and "packet" in rec:
                    rec["packet_id"] = rec["packet"]
                if "mobility_tag" not in rec:
                    rec["mobility_tag"] = "unknown"

                missing = REQUIRED_FIELDS.difference(rec.keys())
                if missing:
                    errors.append(f"{path}:{ln} missing fields: {sorted(missing)}")
                    continue

                seed = int(rec["seed"])
                split = split_name(seed, args.train_max_seed, args.val_max_seed)
                prev = seed_to_split.get(seed)
                if prev is not None and prev != split:
                    errors.append(f"seed leakage: seed={seed} seen in {prev} and {split}")
                    continue
                seed_to_split[seed] = split

                policy = str(rec.get("policy_mode", "unknown"))
                tier = str(rec.get("exploration_tier", "unknown"))
                policy_counts[policy] += 1
                exploration_counts[tier] += 1

                src = {
                    "source_file": str(path),
                    "line": ln,
                    "seed": seed,
                    "policy_mode": policy,
                    "exploration_tier": tier,
                    "reward": float(rec.get("reward", 0.0)),
                }
                splits[split].append(src)

    for policy, count in policy_counts.items():
        if count < args.min_policy_samples:
            errors.append(f"policy support too low: {policy} has {count} samples")

    if exploration_counts.get("high", 0) == 0:
        errors.append("no high exploration samples found (expected epsilon=0.25 episodes)")

    manifest = {
        "input_root": str(in_root),
        "total_lines": total,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "policy_counts": dict(policy_counts),
        "exploration_counts": dict(exploration_counts),
        "seed_count": len(seed_to_split),
        "errors": errors,
    }

    for split, items in splits.items():
        with (out_root / f"{split}_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)

    with (out_root / "dataset_report.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if errors:
        raise SystemExit(
            "Dataset QA failed. See "
            f"{out_root / 'dataset_report.json'} for details ({len(errors)} issue(s))."
        )

    print("Dataset QA passed.")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
