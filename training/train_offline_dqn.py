#!/usr/bin/env python3
"""Offline BS-side DQN training for UAV-assisted WSN routing.

This script trains a centralized Q-network against a digital twin and exports:
1) MLP policy model for direct runtime inference
2) Discretized Q-table for lightweight node-side lookup
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
    raise SystemExit(
        "PyTorch is required for offline DQN training. Install with: pip install torch"
    ) from exc

from digital_twin_env import DigitalTwinRoutingEnv


@dataclass
class Transition:
    state_action: np.ndarray
    reward: float
    next_candidates: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.data: List[Transition] = []
        self.cursor = 0

    def add(self, transition: Transition) -> None:
        if len(self.data) < self.capacity:
            self.data.append(transition)
            return
        self.data[self.cursor] = transition
        self.cursor = (self.cursor + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        idx = self.rng.integers(0, len(self.data), size=batch_size)
        return [self.data[int(i)] for i in idx]

    def __len__(self) -> int:
        return len(self.data)


class RunningStats:
    """Welford running statistics for feature normalization export."""

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


def parse_bins(spec: str) -> List[int]:
    bins = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if not bins:
        raise ValueError("At least one bin is required.")
    if any(v < 2 for v in bins):
        raise ValueError("All bins must be >= 2.")
    return bins


def epsilon_at_step(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return eps_end
    mix = min(1.0, max(0.0, step / decay_steps))
    return eps_start + mix * (eps_end - eps_start)


def choose_action(
    policy: QNetwork,
    candidates: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
) -> int:
    if candidates.shape[0] == 0:
        return 0

    if rng.random() < epsilon:
        return int(rng.integers(0, candidates.shape[0]))

    with torch.no_grad():
        cand_t = torch.tensor(candidates, dtype=torch.float32, device=device)
        q = policy(cand_t).squeeze(-1)
        return int(torch.argmax(q).item())


def optimize_step(
    policy: QNetwork,
    target: QNetwork,
    optimizer: optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> float:
    batch = replay.sample(batch_size)

    sa = np.stack([t.state_action for t in batch], axis=0)
    rewards = np.asarray([t.reward for t in batch], dtype=np.float32)

    sa_t = torch.tensor(sa, dtype=torch.float32, device=device)
    reward_t = torch.tensor(rewards, dtype=torch.float32, device=device)

    pred_q = policy(sa_t).squeeze(-1)

    targets = []
    with torch.no_grad():
        for t in batch:
            y = float(t.reward)
            if (not t.done) and t.next_candidates.shape[0] > 0:
                nxt_t = torch.tensor(t.next_candidates, dtype=torch.float32, device=device)
                y += gamma * float(torch.max(target(nxt_t)).item())
            targets.append(y)
    target_t = torch.tensor(targets, dtype=torch.float32, device=device)

    loss = torch.mean((pred_q - target_t) ** 2)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    return float(loss.item())


def format_float(v: float) -> str:
    return f"{v:.8g}"


def export_mlp_model(model: QNetwork, out_path: Path, mean: np.ndarray, std: np.ndarray) -> None:
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"layers {len(linears)}\n")
        for layer in linears:
            w = layer.weight.detach().cpu().numpy()
            b = layer.bias.detach().cpu().numpy()
            out_dim, in_dim = w.shape

            f.write(f"layer {out_dim} {in_dim}\n")
            f.write("weights ")
            f.write(" ".join(format_float(float(x)) for x in w.reshape(-1)))
            f.write("\n")

            f.write("bias ")
            f.write(" ".join(format_float(float(x)) for x in b.reshape(-1)))
            f.write("\n")

        safe_std = np.maximum(std, 1e-6)
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
        if idx >= b:
            idx = b - 1
        if idx < 0:
            idx = 0
        out.append(idx)
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
    unique_bins = sorted(set(visited_bins))

    entries = []
    with torch.no_grad():
        for idx in unique_bins:
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


def train(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = DigitalTwinRoutingEnv(max_steps=args.max_steps, seed=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    policy = QNetwork(input_dim=8, hidden_dim=args.hidden_dim).to(device)
    target = QNetwork(input_dim=8, hidden_dim=args.hidden_dim).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)
    replay = ReplayBuffer(capacity=args.replay_size, seed=args.seed)

    stats = RunningStats(dim=8)
    visited_bins = set()

    bins = parse_bins(args.qtable_bins)
    if len(bins) != 8:
        raise SystemExit("qtable bins must define exactly 8 dimensions.")

    feature_mins = DigitalTwinRoutingEnv.feature_min.astype(np.float32)
    feature_maxs = DigitalTwinRoutingEnv.feature_max.astype(np.float32)

    global_step = 0
    losses: List[float] = []
    episode_rewards: List[float] = []

    for episode in range(args.episodes):
        candidates = env.reset()
        ep_reward = 0.0

        for _ in range(args.max_steps):
            eps = epsilon_at_step(
                global_step,
                args.epsilon_start,
                args.epsilon_end,
                args.epsilon_decay_steps,
            )

            action_idx = choose_action(policy, candidates, eps, rng, device)
            chosen = candidates[action_idx].astype(np.float32)

            next_candidates, reward, done, _ = env.step(action_idx)

            transition = Transition(
                state_action=chosen,
                reward=float(reward),
                next_candidates=next_candidates.astype(np.float32),
                done=bool(done),
            )
            replay.add(transition)

            stats.update(chosen.astype(np.float64))
            visited_bins.add(discretize(chosen, bins, feature_mins, feature_maxs))
            for row in next_candidates:
                visited_bins.add(discretize(row, bins, feature_mins, feature_maxs))

            ep_reward += reward
            global_step += 1
            candidates = next_candidates

            if len(replay) >= args.batch_size:
                loss = optimize_step(
                    policy=policy,
                    target=target,
                    optimizer=optimizer,
                    replay=replay,
                    batch_size=args.batch_size,
                    gamma=args.gamma,
                    device=device,
                )
                losses.append(loss)

            if global_step % args.target_update_steps == 0:
                target.load_state_dict(policy.state_dict())

            if done:
                break

        episode_rewards.append(ep_reward)

        if (episode + 1) % args.log_interval == 0:
            mean_reward = float(np.mean(episode_rewards[-args.log_interval:]))
            mean_loss = float(np.mean(losses[-args.log_interval:])) if losses else 0.0
            print(
                f"[episode {episode + 1:>7}] "
                f"mean_reward={mean_reward:>8.3f} "
                f"mean_loss={mean_loss:>8.5f} "
                f"buffer={len(replay):>7} "
                f"eps={eps:>5.3f}"
            )

    mean, std = stats.finalize()

    mlp_path = out_dir / args.mlp_name
    qtab_path = out_dir / args.qtable_name
    summary_path = out_dir / args.summary_name

    export_mlp_model(policy, mlp_path, mean, std)
    entries = export_qtable(
        model=policy,
        out_path=qtab_path,
        visited_bins=visited_bins,
        bins=bins,
        mins=feature_mins,
        maxs=feature_maxs,
        device=device,
    )

    summary = {
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "gamma": args.gamma,
        "learning_rate": args.learning_rate,
        "replay_size": args.replay_size,
        "batch_size": args.batch_size,
        "target_update_steps": args.target_update_steps,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay_steps": args.epsilon_decay_steps,
        "global_steps": global_step,
        "final_mean_reward": float(np.mean(episode_rewards[-min(100, len(episode_rewards)):])) if episode_rewards else 0.0,
        "final_mean_loss": float(np.mean(losses[-min(100, len(losses)):])) if losses else 0.0,
        "mlp_model": str(mlp_path),
        "qtable_model": str(qtab_path),
        "qtable_entries": int(entries),
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training complete.")
    print(f"MLP model:   {mlp_path}")
    print(f"Q-table:     {qtab_path}")
    print(f"Summary:     {summary_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline DQN training for UAV-WSN routing")

    p.add_argument("--output-dir", default="models", help="Directory to write trained artifacts")
    p.add_argument("--episodes", type=int, default=20000, help="Number of offline training episodes")
    p.add_argument("--max-steps", type=int, default=240, help="Decision steps per episode")

    p.add_argument("--hidden-dim", type=int, default=64, help="Hidden units per layer")
    p.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    p.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")

    p.add_argument("--replay-size", type=int, default=100000, help="Replay buffer capacity")
    p.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    p.add_argument("--target-update-steps", type=int, default=1000, help="Target-network sync period")

    p.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon")
    p.add_argument("--epsilon-end", type=float, default=0.05, help="Final epsilon")
    p.add_argument("--epsilon-decay-steps", type=int, default=200000, help="Epsilon decay horizon")

    p.add_argument("--qtable-bins", default="10,10,11,10,10,10,10,10", help="Comma-separated discretization bins")

    p.add_argument("--seed", type=int, default=13, help="Random seed")
    p.add_argument("--log-interval", type=int, default=100, help="Episodes between progress prints")
    p.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available")

    p.add_argument("--mlp-name", default="offline_dqn_policy.mdl", help="Exported MLP filename")
    p.add_argument("--qtable-name", default="offline_qtable.qtab", help="Exported Q-table filename")
    p.add_argument("--summary-name", default="offline_training_summary.json", help="Training summary filename")

    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
