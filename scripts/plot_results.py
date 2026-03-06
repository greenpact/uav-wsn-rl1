#!/usr/bin/env python3
"""Generate basic result plots from simulation CSV outputs.

Reads:
- results/packet_generated.csv
- results/packet_delivered.csv
- results/node_deaths.csv

Writes PNG figures under results/plots/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    plots = results / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    gen = safe_read_csv(results / "packet_generated.csv")
    delivered = safe_read_csv(results / "packet_delivered.csv")
    deaths = safe_read_csv(results / "node_deaths.csv")

    if not gen.empty:
        gen = gen.sort_values("t")
        gen["generated_cum"] = range(1, len(gen) + 1)

    if not delivered.empty:
        delivered = delivered.sort_values("t_recv" if "t_recv" in delivered.columns else "t")
        delivered["delivered_cum"] = range(1, len(delivered) + 1)

    if not gen.empty or not delivered.empty:
        plt.figure(figsize=(10, 5))
        if not gen.empty:
            plt.plot(gen["t"], gen["generated_cum"], label="Generated", linewidth=2)
        if not delivered.empty:
            xcol = "t_recv" if "t_recv" in delivered.columns else "t"
            plt.plot(delivered[xcol], delivered["delivered_cum"], label="Delivered", linewidth=2)
        plt.xlabel("Simulation Time (s)")
        plt.ylabel("Cumulative Packets")
        plt.title("Packet Generation vs Delivery")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots / "packet_flow.png", dpi=150)
        plt.close()

    if not delivered.empty and "delay" in delivered.columns:
        plt.figure(figsize=(10, 5))
        plt.hist(delivered["delay"].dropna(), bins=30)
        plt.xlabel("Delay (s)")
        plt.ylabel("Count")
        plt.title("End-to-End Delay Distribution")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots / "delay_histogram.png", dpi=150)
        plt.close()

    if not deaths.empty and "t_death" in deaths.columns:
        deaths = deaths.sort_values("t_death")
        deaths["dead_nodes_cum"] = range(1, len(deaths) + 1)
        plt.figure(figsize=(10, 5))
        plt.step(deaths["t_death"], deaths["dead_nodes_cum"], where="post")
        plt.xlabel("Simulation Time (s)")
        plt.ylabel("Cumulative Dead Nodes")
        plt.title("Node Death Timeline")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots / "node_deaths.png", dpi=150)
        plt.close()

    print(f"Plots generated in: {plots}")


if __name__ == "__main__":
    main()
