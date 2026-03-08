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
import numpy as np
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
        fig, ax_gen = plt.subplots(figsize=(10, 5))

        if not gen.empty:
            ax_gen.plot(
                gen["t"],
                gen["generated_cum"],
                label="Generated (left axis)",
                linewidth=2,
                color="#1f77b4",
            )

        ax_gen.set_xlabel("Simulation Time (s)")
        ax_gen.set_ylabel("Generated Cumulative Packets", color="#1f77b4")
        ax_gen.tick_params(axis="y", labelcolor="#1f77b4")

        delivered_count = 0
        generated_count = 0
        if not gen.empty:
            generated_count = int(gen["generated_cum"].iloc[-1])

        ax_del = ax_gen.twinx()
        if not delivered.empty:
            xcol = "t_recv" if "t_recv" in delivered.columns else "t"
            ax_del.plot(
                delivered[xcol],
                delivered["delivered_cum"],
                label="Delivered (right axis)",
                linewidth=2,
                marker="o",
                markersize=2,
                color="#d62728",
            )
            delivered_count = int(delivered["delivered_cum"].iloc[-1])

        ax_del.set_ylabel("Delivered Cumulative Packets", color="#d62728")
        ax_del.tick_params(axis="y", labelcolor="#d62728")

        pdr = (delivered_count / generated_count) if generated_count > 0 else 0.0
        ax_gen.set_title("Packet Generation vs Delivery")
        ax_gen.text(
            0.01,
            0.98,
            f"Generated={generated_count}  Delivered={delivered_count}  PDR={pdr:.6f}",
            transform=ax_gen.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

        lines1, labels1 = ax_gen.get_legend_handles_labels()
        lines2, labels2 = ax_del.get_legend_handles_labels()
        ax_gen.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        ax_gen.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots / "packet_flow.png", dpi=150)
        plt.close(fig)

    if not gen.empty:
        g_times = gen["t"].dropna().to_numpy(dtype=float)
        d_times = np.array([], dtype=float)
        if not delivered.empty:
            dcol = "t_recv" if "t_recv" in delivered.columns else "t"
            d_times = delivered[dcol].dropna().to_numpy(dtype=float)

        t_max = float(np.max(g_times)) if g_times.size > 0 else 0.0
        if d_times.size > 0:
            t_max = max(t_max, float(np.max(d_times)))
        if t_max <= 0:
            t_max = 1.0

        bins = np.linspace(0.0, t_max, 41)
        plt.figure(figsize=(10, 5))
        plt.hist(
            [g_times, d_times],
            bins=bins,
            label=["Generated", "Delivered"],
            color=["#1f77b4", "#d62728"],
            alpha=0.65,
        )
        plt.xlabel("Simulation Time (s)")
        plt.ylabel("Packets Per Time Bin")
        plt.title("Generated vs Delivered Packets (Histogram)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots / "packet_histogram_generated_vs_delivered.png", dpi=150)
        plt.close()

        # Continuous cumulative PDR(t) = delivered_up_to_t / generated_up_to_t
        g_sorted = np.sort(g_times)
        d_sorted = np.sort(d_times) if d_times.size > 0 else np.array([], dtype=float)
        t_grid = np.linspace(0.0, t_max, 500)
        g_cum = np.searchsorted(g_sorted, t_grid, side="right")
        d_cum = np.searchsorted(d_sorted, t_grid, side="right")
        pdr_t = np.divide(
            d_cum,
            g_cum,
            out=np.zeros_like(t_grid, dtype=float),
            where=g_cum > 0,
        )

        plt.figure(figsize=(10, 5))
        plt.plot(t_grid, pdr_t, linewidth=2, color="#2ca02c")
        plt.xlabel("Simulation Time (s)")
        plt.ylabel("Cumulative PDR")
        plt.title("Continuous PDR Over Simulation Time")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots / "pdr_over_time.png", dpi=150)
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
