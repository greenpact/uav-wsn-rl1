#!/usr/bin/env python3
"""Plot initial network deployment layout (sensors, BS, UAV, and field bounds).

Primary source:
- results/node_positions.csv (generated during simulation initialization)

Fallback source:
- omnetpp.ini values + deterministic random sampling
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_omnetpp_ini(path: Path) -> dict[str, float]:
    vals: dict[str, float] = {
        "numNodes": 400.0,
        "areaX": 1000.0,
        "areaY": 1000.0,
        "bsX": -200.0,
        "bsY": 500.0,
        "uavX": -200.0,
        "uavY": 500.0,
    }

    key_map = {
        "*.numNodes": "numNodes",
        "*.areaX": "areaX",
        "*.areaY": "areaY",
        "*.bs.initialX": "bsX",
        "*.bs.initialY": "bsY",
        "*.uav.initialX": "uavX",
        "*.uav.initialY": "uavY",
    }

    if not path.exists():
        return vals

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
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


def load_positions(results_dir: Path, ini_vals: dict[str, float]) -> tuple[pd.DataFrame, float, float, tuple[float, float], tuple[float, float], bool]:
    node_pos = results_dir / "node_positions.csv"

    area_x = float(ini_vals["areaX"])
    area_y = float(ini_vals["areaY"])
    bs = (float(ini_vals["bsX"]), float(ini_vals["bsY"]))
    uav = (float(ini_vals["uavX"]), float(ini_vals["uavY"]))

    if node_pos.exists() and node_pos.stat().st_size > 0:
        df = pd.read_csv(node_pos)
        sensors = df[df["role"] == "sensor"].copy()
        bs_rows = df[df["role"] == "base_station"]
        if not bs_rows.empty:
            bs = (float(bs_rows.iloc[0]["x"]), float(bs_rows.iloc[0]["y"]))
        return sensors, area_x, area_y, bs, uav, False

    # Fallback: synthetic deterministic deployment from scenario bounds.
    n = int(ini_vals["numNodes"])
    rng = np.random.default_rng(0)
    sensors = pd.DataFrame(
        {
            "x": rng.uniform(0.0, area_x, size=n),
            "y": rng.uniform(0.0, area_y, size=n),
        }
    )
    return sensors, area_x, area_y, bs, uav, True


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    plots = results / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    ini_vals = parse_omnetpp_ini(root / "omnetpp.ini")
    sensors, area_x, area_y, bs, uav, used_fallback = load_positions(results, ini_vals)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Network field rectangle.
    field = plt.Rectangle((0.0, 0.0), area_x, area_y, fill=False, linewidth=2.0, linestyle="--", color="black")
    ax.add_patch(field)

    if not sensors.empty:
        ax.scatter(
            sensors["x"],
            sensors["y"],
            s=14,
            alpha=0.7,
            c="#1f77b4",
            edgecolors="none",
            label=f"Sensors (n={len(sensors)})",
        )

    ax.scatter([bs[0]], [bs[1]], s=140, marker="^", c="#d62728", label="Base Station")
    ax.scatter([uav[0]], [uav[1]], s=140, marker="*", c="#2ca02c", label="UAV (initial)")

    pad_x = max(100.0, 0.1 * area_x)
    pad_y = max(100.0, 0.1 * area_y)
    ax.set_xlim(min(-250.0, bs[0] - pad_x), area_x + pad_x)
    ax.set_ylim(min(-100.0, bs[1] - pad_y), area_y + pad_y)

    subtitle = "actual positions from results/node_positions.csv"
    if used_fallback:
        subtitle = "fallback synthetic positions (node_positions.csv not found)"

    ax.set_title(f"Initial Network Deployment Layout\n({subtitle})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out = plots / "network_layout_initial_deployment.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)

    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
