#!/usr/bin/env python3
"""Train offline DQN-style scorer from fixed OMNeT++ transition datasets.

Consumes JSONL transition logs produced by TrlarRouting and exports:
- MLP model (.mdl)
- discretized Q-table (.qtab)
- training summary JSON
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required. Install with: pip install torch") from exc


@dataclass
class Sample:
    state_action: np.ndarray
    reward: float
    next_candidates: np.ndarray
    done: bool


class QNetwork(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RunningStats:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.count < 2:
            return self.mean.astype(np.float32), np.ones(self.dim, dtype=np.float32)
        var = self.m2 / (self.count - 1)
        std = np.sqrt(np.maximum(var, 1e-12))
        return self.mean.astype(np.float32), std.astype(np.float32)


def parse_bins(spec: str) -> List[int]:
    bins = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if len(bins) != 8:
        raise ValueError("qtable bins must define exactly 8 dimensions")
    if any(v < 2 for v in bins):
        raise ValueError("all bins must be >= 2")
    return bins


def format_float(v: float) -> str:
    return f"{v:.8g}"


def export_mlp_model(model: QNetwork, out_path: Path, mean: np.ndarray, std: np.ndarray) -> None:
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    safe_std = np.maximum(std, 1e-6)

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"layers {len(linears)}\n")
        for layer in linears:
            w = layer.weight.detach().cpu().numpy()
            b = layer.bias.detach().cpu().numpy()
            out_dim, in_dim = w.shape
            f.write(f"layer {out_dim} {in_dim}\n")
            f.write("weights " + " ".join(format_float(float(x)) for x in w.reshape(-1)) + "\n")
            f.write("bias " + " ".join(format_float(float(x)) for x in b.reshape(-1)) + "\n")
        f.write(f"norm {mean.size}\n")
        f.write("mean " + " ".join(format_float(float(x)) for x in mean.tolist()) + "\n")
        f.write("std " + " ".join(format_float(float(x)) for x in safe_std.tolist()) + "\n")


def discretize(features: np.ndarray, bins: Sequence[int], mins: np.ndarray, maxs: np.ndarray) -> Tuple[int, ...]:
    out = []
    for i, b in enumerate(bins):
        lo = float(mins[i])
        hi = float(maxs[i])
        if hi <= lo:
            out.append(0)
            continue
        v = float(np.clip(features[i], lo, hi))
        ratio = (v - lo) / (hi - lo)
        idx = int(math.floor(ratio * b))
        out.append(min(max(idx, 0), b - 1))
    return tuple(out)


def bin_center(index: Sequence[int], bins: Sequence[int], mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    center = np.zeros(len(index), dtype=np.float32)
    for i, idx in enumerate(index):
        lo = float(mins[i])
        hi = float(maxs[i])
        width = (hi - lo) / float(bins[i])
        center[i] = np.float32(lo + (idx + 0.5) * width)
    return center


def export_qtable(
    model: QNetwork,
    out_path: Path,
    visited_bins: Iterable[Tuple[int, ...]],
    bins: Sequence[int],
    mins: np.ndarray,
    maxs: np.ndarray,
    device: torch.device,
    default_q: float = 0.0,
) -> int:
    entries = []
    with torch.no_grad():
        for idx in sorted(set(visited_bins)):
            feat = bin_center(idx, bins, mins, maxs)
            q = float(model(torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)).item())
            entries.append((idx, q))

    with out_path.open("w", encoding="utf-8") as f:
        f.write("format qtable_v1\n")
        f.write(f"dims {len(bins)}\n")
        f.write("bins " + " ".join(str(int(x)) for x in bins) + "\n")
        f.write("state_min " + " ".join(format_float(float(x)) for x in mins.tolist()) + "\n")
        f.write("state_max " + " ".join(format_float(float(x)) for x in maxs.tolist()) + "\n")
        f.write(f"default {format_float(default_q)}\n")
        f.write(f"entries {len(entries)}\n")
        for idx, q in entries:
            f.write(" ".join(str(int(v)) for v in idx) + " " + format_float(q) + "\n")

    return len(entries)


def load_transition_samples(input_root: Path, max_samples: int, min_reward: float) -> list[Sample]:
    files = sorted(input_root.glob("run-*/transitions*.jsonl"))
    if not files:
        files = sorted(input_root.glob("run-*/transitions.jsonl"))
    if not files:
        raise SystemExit(f"No transition files found under {input_root}")

    samples: list[Sample] = []
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") != "transition":
                    continue

                action = rec.get("action_features")
                nxt = rec.get("next_candidates", [])
                reward = float(rec.get("reward", 0.0))
                done = bool(rec.get("done", False))

                if not isinstance(action, list) or len(action) != 8:
                    continue
                if reward < min_reward:
                    continue

                next_candidates = []
                if isinstance(nxt, list):
                    for c in nxt:
                        if isinstance(c, list) and len(c) == 8:
                            next_candidates.append(c)

                samples.append(
                    Sample(
                        state_action=np.asarray(action, dtype=np.float32),
                        reward=reward,
                        next_candidates=np.asarray(next_candidates, dtype=np.float32)
                        if next_candidates
                        else np.zeros((0, 8), dtype=np.float32),
                        done=done,
                    )
                )
                if max_samples > 0 and len(samples) >= max_samples:
                    return samples

    if not samples:
        raise SystemExit("No usable transition samples found.")
    return samples


def optimize_batch(
    policy: QNetwork,
    target: QNetwork,
    optimizer: optim.Optimizer,
    batch: list[Sample],
    gamma: float,
    device: torch.device,
) -> float:
    sa = np.stack([s.state_action for s in batch], axis=0)
    rewards = np.asarray([s.reward for s in batch], dtype=np.float32)

    sa_t = torch.tensor(sa, dtype=torch.float32, device=device)
    pred_q = policy(sa_t).squeeze(-1)

    with torch.no_grad():
        yvals = []
        for s in batch:
            y = float(s.reward)
            if (not s.done) and s.next_candidates.shape[0] > 0:
                nc = torch.tensor(s.next_candidates, dtype=torch.float32, device=device)
                y += gamma * float(torch.max(target(nc)).item())
            yvals.append(y)
        y_t = torch.tensor(yvals, dtype=torch.float32, device=device)

    loss = torch.mean((pred_q - y_t) ** 2)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train offline DQN from fixed transition dataset")
    p.add_argument("--input", required=True, help="Root with run-*/transitions*.jsonl")
    p.add_argument("--output-dir", default="models", help="Output directory")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--target-sync-steps", type=int, default=500)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--max-samples", type=int, default=0, help="0 means no cap")
    p.add_argument("--min-reward", type=float, default=-1e9)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--qtable-bins", default="10,10,11,10,10,10,10,10")
    p.add_argument("--mlp-name", default="offline_dqn_policy.mdl")
    p.add_argument("--qtable-name", default="offline_qtable.qtab")
    p.add_argument("--summary-name", default="offline_training_summary.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    input_root = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_transition_samples(input_root, args.max_samples, args.min_reward)
    print(f"Loaded {len(samples)} transition samples")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    policy = QNetwork(input_dim=8, hidden_dim=args.hidden_dim).to(device)
    target = QNetwork(input_dim=8, hidden_dim=args.hidden_dim).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)

    stats = RunningStats(dim=8)
    visited_bins: set[Tuple[int, ...]] = set()

    mins = np.zeros(8, dtype=np.float32)
    maxs = np.ones(8, dtype=np.float32)
    bins = parse_bins(args.qtable_bins)

    for s in samples:
        stats.update(s.state_action.astype(np.float64))
        visited_bins.add(discretize(s.state_action, bins, mins, maxs))
        for nxt in s.next_candidates:
            visited_bins.add(discretize(nxt, bins, mins, maxs))

    steps = 0
    losses: list[float] = []
    for epoch in range(args.epochs):
        order = rng.permutation(len(samples))
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            batch = [samples[int(i)] for i in idx]
            loss = optimize_batch(policy, target, optimizer, batch, args.gamma, device)
            losses.append(loss)
            steps += 1
            if steps % args.target_sync_steps == 0:
                target.load_state_dict(policy.state_dict())

        mean_loss = float(np.mean(losses[-max(1, len(order) // max(1, args.batch_size)):]))
        print(f"[epoch {epoch + 1:03d}] mean_loss={mean_loss:.6f}")

    mean, std = stats.finalize()

    mlp_path = out_dir / args.mlp_name
    qtab_path = out_dir / args.qtable_name
    summary_path = out_dir / args.summary_name

    export_mlp_model(policy, mlp_path, mean, std)
    entries = export_qtable(policy, qtab_path, visited_bins, bins, mins, maxs, device)

    summary = {
        "input_root": str(input_root),
        "sample_count": len(samples),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "target_sync_steps": args.target_sync_steps,
        "hidden_dim": args.hidden_dim,
        "mean_loss_last": float(np.mean(losses[-min(100, len(losses)):])) if losses else 0.0,
        "mlp_model": str(mlp_path),
        "qtable_model": str(qtab_path),
        "qtable_entries": int(entries),
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training complete.")
    print(f"MLP model: {mlp_path}")
    print(f"Q-table: {qtab_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
