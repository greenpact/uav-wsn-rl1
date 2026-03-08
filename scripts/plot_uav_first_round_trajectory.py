#!/usr/bin/env python3
"""Plot UAV first-round trajectory for General configuration.

The path is sampled from the same motion logic implemented in src/uav/UavModule.cc.
Output:
- results/plots/uav_first_round_trajectory_general.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_general_params(ini_path: Path) -> dict[str, float]:
    vals = {
        "areaX": 1000.0,
        "areaY": 1000.0,
        "bsX": -200.0,
        "bsY": 500.0,
        "initialX": -200.0,
        "initialY": 500.0,
        "missionPeriod": 280.0,
        "contactWindow": 8.0,
    }

    key_map = {
        "*.areaX": "areaX",
        "*.areaY": "areaY",
        "*.bs.initialX": "bsX",
        "*.bs.initialY": "bsY",
        "*.uav.initialX": "initialX",
        "*.uav.initialY": "initialY",
        "*.uav.missionPeriod": "missionPeriod",
        "*.uav.contactWindow": "contactWindow",
    }

    if not ini_path.exists():
        return vals

    in_general = False
    for raw in ini_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_general = line == "[General]"
            continue
        if not in_general:
            continue
        if "=" not in line:
            continue
        lhs, rhs = [x.strip() for x in line.split("=", 1)]
        if lhs in key_map:
            try:
                vals[key_map[lhs]] = float(rhs)
            except ValueError:
                pass

    return vals


def uav_xy(t: float, p: dict[str, float]) -> tuple[float, float]:
    area_x = p["areaX"]
    area_y = p["areaY"]
    bs_x = p["bsX"]
    bs_y = p["bsY"]
    cycle = max(30.0, p["missionPeriod"])
    round_index = int(math.floor(max(0.0, t) / cycle))
    phase = max(0.0, t) - round_index * cycle

    def unit_noise(ridx: int, salt: int) -> float:
        xx = math.sin((ridx + 1) * (12.9898 + salt) + 78.233 + 0.123 * salt) * 43758.5453
        return xx - math.floor(xx)

    hold = max(2.0, min(p["contactWindow"], 0.20 * cycle))
    travel_in = max(4.0, 0.18 * cycle)
    travel_out = max(4.0, 0.18 * cycle)
    survey = max(2.0, cycle - hold - travel_in - travel_out)

    entry_y = area_y * unit_noise(round_index, 11)
    exit_y = area_y * unit_noise(round_index, 29)
    curve_phase = 2.0 * math.pi * unit_noise(round_index, 47)
    curve_mix = unit_noise(round_index, 73)

    if phase < hold:
        x = bs_x
        y = bs_y
    elif phase < hold + travel_in:
        alpha = (phase - hold) / max(1e-9, travel_in)
        x = bs_x + alpha * (0.0 - bs_x)
        y = bs_y + alpha * (entry_y - bs_y)
    elif phase < hold + travel_in + survey:
        u = (phase - hold - travel_in) / max(1e-9, survey)
        base_x = area_x * u
        base_y = entry_y + (exit_y - entry_y) * u
        amp_y = 0.18 * area_y + curve_mix * 0.22 * area_y
        wiggle = math.sin((2.0 * math.pi * (1.0 + curve_mix)) * u + curve_phase)
        x = base_x
        y = base_y + amp_y * wiggle
    elif phase < hold + travel_in + survey + travel_out:
        alpha = (phase - hold - travel_in - survey) / max(1e-9, travel_out)
        x = area_x + alpha * (bs_x - area_x)
        y = exit_y + alpha * (bs_y - exit_y)
    else:
        x = bs_x
        y = bs_y

    x = max(-250.0, min(area_x + 250.0, x))
    y = max(-250.0, min(area_y + 250.0, y))
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot UAV trajectory for a selected mission round")
    parser.add_argument(
        "--round",
        type=int,
        default=1,
        help="1-based mission round index (e.g., 1 for first, 10 for tenth)",
    )
    args = parser.parse_args()

    round_index = max(1, args.round)

    root = Path(__file__).resolve().parents[1]
    ini_path = root / "omnetpp.ini"
    plots_dir = root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    p = parse_general_params(ini_path)
    period = max(30.0, p["missionPeriod"])

    t0 = (round_index - 1) * period
    t1 = round_index * period
    t = np.linspace(t0, t1, 1200)
    xy = np.array([uav_xy(float(tt), p) for tt in t])

    bs = np.array([p["bsX"], p["bsY"]])
    start = xy[0]
    end = xy[-1]

    fig, ax = plt.subplots(figsize=(10, 8))

    field = plt.Rectangle((0.0, 0.0), p["areaX"], p["areaY"], fill=False, linewidth=2.0, linestyle="--", color="black")
    ax.add_patch(field)

    ax.plot(xy[:, 0], xy[:, 1], color="#1f77b4", linewidth=2.2, label="UAV first-round trajectory")

    # Direction markers along the path.
    idx = np.linspace(0, len(t) - 1, 16, dtype=int)
    ax.scatter(xy[idx, 0], xy[idx, 1], s=12, color="#1f77b4", alpha=0.8)

    ax.scatter([bs[0]], [bs[1]], s=170, marker="^", c="#d62728", label="Base Station (BS)")
    ax.scatter([start[0]], [start[1]], s=120, marker="o", c="#2ca02c", label=f"Start (t={t0:.0f}s)")
    ax.scatter([end[0]], [end[1]], s=120, marker="X", c="#9467bd", label=f"End (t={t1:.0f}s)")

    ax.annotate("Start", (start[0], start[1]), textcoords="offset points", xytext=(8, 8))
    ax.annotate("End (BS)", (end[0], end[1]), textcoords="offset points", xytext=(8, -14))

    pad_x = max(100.0, 0.1 * p["areaX"])
    pad_y = max(100.0, 0.1 * p["areaY"])
    ax.set_xlim(min(-250.0, p["bsX"] - pad_x), p["areaX"] + pad_x)
    ax.set_ylim(min(-100.0, p["bsY"] - pad_y), p["areaY"] + pad_y)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(
        f"General Config: UAV Round {round_index} Trajectory "
        f"(BS -> Area -> BS)"
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()

    out = plots_dir / f"uav_round_{round_index}_trajectory_general.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
