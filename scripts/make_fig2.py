#!/usr/bin/env python3
"""Regenerate Fig. 2 with v19 original-response terminology."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = Path(__file__).with_name("fig2_rank.py")
SPEC = importlib.util.spec_from_file_location("grsl_fig_v4", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import base figure module: {BASE_PATH}")
FIG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIG)


def panel_geometry(ax, rows: list[dict[str, str]]) -> None:
    base = FIG.BASE
    rng = np.random.default_rng(20260718)
    positions: list[float] = []
    values: list[np.ndarray] = []
    labels: list[str] = []
    facecolors: list[str] = []
    position = 1.0
    short_labels = {"dense_narrow": "Dense", "reference": "Orig.", "coarse_broad": "Coarse"}
    for geometry in base.GEOMETRIES:
        for direction in ("minus", "plus"):
            subset = [
                row
                for row in rows
                if row["geometry"] == geometry and row["direction"] == direction
            ]
            if len(subset) != 18:
                raise ValueError(f"Expected 18 rows for {geometry}/{direction}, got {len(subset)}")
            drop = np.asarray([100.0 * float(row["paired_macro_f1_drop_mean"]) for row in subset])
            positions.append(position)
            values.append(drop)
            labels.append(f"{short_labels[geometry]}\n{'-' if direction == 'minus' else '+'}")
            facecolors.append("#DCEAF4" if geometry == "reference" else "#EEEEEE")
            for row, value in zip(subset, drop):
                ax.scatter(
                    position + rng.uniform(-0.13, 0.13),
                    value,
                    s=13,
                    marker=base.DIRECTION_MARKERS[direction],
                    facecolor=base.SCENE_COLORS[row["scene"]],
                    edgecolor="white",
                    linewidth=0.25,
                    alpha=0.78,
                    zorder=3,
                )
            position += 1.0
        position += 0.35
    boxes = ax.boxplot(
        values,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 1.2},
        whiskerprops={"color": "#666666", "linewidth": 0.7},
        capprops={"color": "#666666", "linewidth": 0.7},
        boxprops={"color": "#666666", "linewidth": 0.7},
    )
    for patch, color in zip(boxes["boxes"], facecolors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    ax.axhline(0.0, color="#555555", linestyle=":", linewidth=0.8)
    ax.set_title("(d)  Virtual-response sensitivity", loc="left", fontweight="bold")
    ax.set_ylabel("Macro-F1 drop (percentage points)")
    ax.set_xticks(positions, labels)
    ax.tick_params(axis="x", labelsize=8.0)
    ax.set_xlim(0.45, positions[-1] + 0.55)
    ax.set_ylim(bottom=-1.0)
    base.style_axis(ax)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dose", type=Path, default=ROOT / "data/reference_panel/dose_monotonicity.csv")
    parser.add_argument("--nominal-reliability", type=Path, default=ROOT / "outputs/analysis/nominal_reliability_cells.csv")
    parser.add_argument("--primary", type=Path, default=ROOT / "data/reference_panel/primary_contrasts.csv")
    parser.add_argument("--geometry", type=Path, default=ROOT / "data/geometry_panel/geometry_primary_results.csv")
    parser.add_argument("--probabilistic", type=Path, default=ROOT / "data/reference_secondary/primary_probabilistic_metrics.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/Fig2_sensitivity_audit.pdf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    FIG.BASE.panel_geometry = panel_geometry
    figure = FIG.build_figure(
        FIG.BASE.read_csv(args.dose),
        FIG.BASE.read_csv(args.nominal_reliability),
        FIG.BASE.read_csv(args.primary),
        FIG.BASE.read_csv(args.geometry),
        FIG.BASE.read_csv(args.probabilistic),
    )
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=400, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(output.with_suffix(".png"), dpi=500, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print(output)

if __name__ == "__main__":
    main()
