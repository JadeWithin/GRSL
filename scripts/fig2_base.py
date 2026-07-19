#!/usr/bin/env python3
"""Generate the complete four-panel GRSL evidence figure from derived CSV files.

This script is intentionally self-contained. It replaces the v2 wrapper whose
base plotting module was not included in the clean manuscript package.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
SCENES = ("LongKou", "HanChuan", "HongHu")
SCENE_COLORS = {
    "LongKou": "#0072B2",
    "HanChuan": "#D55E00",
    "HongHu": "#009E73",
}
GEOMETRIES = ("dense_narrow", "reference", "coarse_broad")
GEOMETRY_LABELS = {
    "dense_narrow": "Dense/narrow",
    "reference": "Original",
    "coarse_broad": "Coarse/broad",
}
DIRECTION_MARKERS = {"minus": "o", "plus": "^"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def style_axis(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.75)
    ax.tick_params(length=2.5, width=0.6, pad=1.8)


def panel_dose(ax: mpl.axes.Axes, rows: list[dict[str, str]]) -> None:
    doses = np.asarray([0.0, 1.1, 2.2, 4.4])
    columns = ("drop_0p0_pp", "drop_1p1_pp", "drop_2p2_pp", "drop_4p4_pp")
    grouped: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for row in rows:
        values = np.asarray([float(row[column]) for column in columns])
        scene = row["scene"]
        direction = row["direction"]
        grouped[(scene, direction)].append(values)
        ax.plot(
            doses,
            values,
            color=SCENE_COLORS[scene],
            linewidth=0.55,
            alpha=0.28,
            marker=DIRECTION_MARKERS[direction],
            markersize=1.9,
            markeredgewidth=0,
        )
    for (scene, direction), curves in grouped.items():
        median = np.median(np.stack(curves), axis=0)
        ax.plot(
            doses,
            median,
            color=SCENE_COLORS[scene],
            linewidth=1.8,
            linestyle="-" if direction == "minus" else "--",
            marker=DIRECTION_MARKERS[direction],
            markersize=3.0,
            markeredgecolor="white",
            markeredgewidth=0.35,
            zorder=4,
        )
    ax.set_title("(a)  Directional dose response", loc="left", fontweight="bold")
    ax.set_xlabel(r"Absolute center shift $|\Delta\lambda|$ (nm)")
    ax.set_ylabel("Macro-F1 drop (percentage points)")
    ax.set_xticks(doses)
    ax.set_xlim(-0.15, 4.6)
    ax.set_ylim(bottom=-1.0)
    style_axis(ax)


def panel_transitions(ax: mpl.axes.Axes, rows: list[dict[str, str]]) -> None:
    maximum = 0.0
    for row in rows:
        scene = row["scene"]
        direction = row["direction"]
        wrong_to_correct = 100.0 * float(row["wrong_to_correct_rate_mean"])
        correct_to_wrong = 100.0 * float(row["correct_to_wrong_rate_mean"])
        flip = 100.0 * float(row["label_flip_rate_mean"])
        maximum = max(maximum, wrong_to_correct, correct_to_wrong)
        ax.scatter(
            wrong_to_correct,
            correct_to_wrong,
            s=12.0 + 2.2 * flip,
            marker=DIRECTION_MARKERS[direction],
            facecolor=SCENE_COLORS[scene],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.82,
            zorder=3,
        )
    limit = np.ceil((maximum + 1.0) / 5.0) * 5.0
    ax.plot([0, limit], [0, limit], color="#555555", linestyle=":", linewidth=0.8)
    ax.set_xlim(-0.7, limit)
    ax.set_ylim(-0.7, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("(b)  Original-response transitions", loc="left", fontweight="bold")
    ax.set_xlabel(r"W$\rightarrow$C (% test pixels)")
    ax.set_ylabel(r"C$\rightarrow$W (% test pixels)")
    style_axis(ax)


def panel_geometry(ax: mpl.axes.Axes, rows: list[dict[str, str]]) -> None:
    rng = np.random.default_rng(20260718)
    positions: list[float] = []
    values: list[np.ndarray] = []
    labels: list[str] = []
    facecolors: list[str] = []
    position = 1.0
    for geometry in GEOMETRIES:
        for direction in ("minus", "plus"):
            subset = [
                row
                for row in rows
                if row["geometry"] == geometry and row["direction"] == direction
            ]
            if len(subset) != 18:
                raise ValueError(f"Expected 18 rows for {geometry}/{direction}, got {len(subset)}")
            drop = np.asarray(
                [100.0 * float(row["paired_macro_f1_drop_mean"]) for row in subset]
            )
            positions.append(position)
            values.append(drop)
            short_geometry = {'dense_narrow': 'Dense', 'reference': 'Ref.', 'coarse_broad': 'Coarse'}[geometry]
            labels.append(f"{short_geometry}\n{'-' if direction == 'minus' else '+'}")
            facecolors.append("#DCEAF4" if geometry == "reference" else "#EEEEEE")
            for row, value in zip(subset, drop):
                ax.scatter(
                    position + rng.uniform(-0.13, 0.13),
                    value,
                    s=13,
                    marker=DIRECTION_MARKERS[direction],
                    facecolor=SCENE_COLORS[row["scene"]],
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
    ax.set_title("(c)  Virtual-response sensitivity", loc="left", fontweight="bold")
    ax.set_ylabel("Macro-F1 drop (percentage points)")
    ax.set_xticks(positions, labels)
    ax.tick_params(axis="x", labelsize=6.0)
    ax.set_xlim(0.45, positions[-1] + 0.55)
    ax.set_ylim(bottom=-1.0)
    style_axis(ax)


def calibration_strip(
    ax: mpl.axes.Axes,
    rows: list[dict[str, str]],
    column: str,
    title: str,
    scale: float,
    positive_count: int,
) -> None:
    rng = np.random.default_rng(20260718 + len(title))
    data = np.asarray([scale * float(row[column]) for row in rows])
    ax.boxplot(
        [data],
        positions=[1.0],
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 1.1},
        boxprops={"facecolor": "#ECECEC", "edgecolor": "#666666", "linewidth": 0.7},
        whiskerprops={"color": "#666666", "linewidth": 0.7},
        capprops={"color": "#666666", "linewidth": 0.7},
    )
    for row, value in zip(rows, data):
        ax.scatter(
            1.0 + rng.uniform(-0.14, 0.14),
            value,
            s=12,
            marker=DIRECTION_MARKERS[row["direction"]],
            facecolor=SCENE_COLORS[row["scene"]],
            edgecolor="white",
            linewidth=0.25,
            alpha=0.78,
            zorder=3,
        )
    ax.axhline(0.0, color="#555555", linestyle=":", linewidth=0.8)
    ax.set_xlim(0.62, 1.38)
    ax.set_xticks([])
    ax.set_title(title, fontsize=8.4, pad=2.5)
    ax.text(
        0.5,
        0.98,
        f"{positive_count}/36 > 0",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.8,
        color="#333333",
    )
    style_axis(ax)


def build_figure(
    dose_rows: list[dict[str, str]],
    primary_rows: list[dict[str, str]],
    geometry_rows: list[dict[str, str]],
    probabilistic_rows: list[dict[str, str]],
) -> mpl.figure.Figure:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.16, 4.75), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        2,
        left=0.072,
        right=0.992,
        bottom=0.095,
        top=0.975,
        wspace=0.28,
        hspace=0.42,
    )
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    ax_c = fig.add_subplot(outer[1, 0])
    panel_dose(ax_a, dose_rows)
    panel_transitions(ax_b, primary_rows)
    panel_geometry(ax_c, geometry_rows)

    inner = outer[1, 1].subgridspec(1, 3, wspace=0.60)
    calibration_axes = [fig.add_subplot(inner[0, index]) for index in range(3)]
    calibration_strip(calibration_axes[0], probabilistic_rows, "nll_change", r"$\Delta$NLL", 1.0, 36)
    calibration_strip(calibration_axes[1], probabilistic_rows, "brier_change", r"$\Delta$Brier (pp)", 100.0, 35)
    calibration_strip(calibration_axes[2], probabilistic_rows, "ece_change", r"$\Delta$ECE (pp)", 100.0, 31)
    calibration_axes[0].text(
        -0.30,
        1.13,
        "(d)  Original-response probabilistic metrics",
        transform=calibration_axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
    )

    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=SCENE_COLORS[scene],
            markeredgecolor="none", markersize=4.2, label=scene
        )
        for scene in SCENES
    ]
    legend_handles.extend(
        [
            Line2D([0], [0], marker="o", color="#555555", linestyle="none", markersize=4.0, label=r"$-2.2$ nm"),
            Line2D([0], [0], marker="^", color="#555555", linestyle="none", markersize=4.0, label=r"$+2.2$ nm"),
        ]
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.74, 0.004),
        ncol=5,
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dose", type=Path, default=ROOT / "data/reference_panel/dose_monotonicity.csv")
    parser.add_argument("--primary", type=Path, default=ROOT / "data/reference_panel/primary_contrasts.csv")
    parser.add_argument("--geometry", type=Path, default=ROOT / "data/geometry_panel/geometry_primary_results.csv")
    parser.add_argument("--probabilistic", type=Path, default=ROOT / "data/reference_secondary/primary_probabilistic_metrics.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "assets/images/Fig2_strategy_a_geometry_complete_v12.pdf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figure = build_figure(
        read_csv(args.dose),
        read_csv(args.primary),
        read_csv(args.geometry),
        read_csv(args.probabilistic),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(args.output.with_suffix(".png"), dpi=500, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


if __name__ == "__main__":
    main()
