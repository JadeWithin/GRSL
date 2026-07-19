#!/usr/bin/env python3
"""Generate the five-panel GRSL evidence figure with rank discordance."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = Path(__file__).with_name("fig2_base.py")
SPEC = importlib.util.spec_from_file_location("grsl_fig_v3", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import base figure module: {BASE_PATH}")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

SCENE_JITTER = {"LongKou": -0.10, "HanChuan": 0.0, "HongHu": 0.10}


def panel_nominal_reliability(
    ax: mpl.axes.Axes, rows: list[dict[str, str]]
) -> None:
    """Compare within-scene nominal and mean-loss reliability ranks."""
    for row in rows:
        scene = row["scene"]
        jitter = SCENE_JITTER[scene]
        ax.scatter(
            float(row["nominal_rank"]) + jitter,
            float(row["mean_loss_reliability_rank"]) + jitter,
            s=24,
            marker="o",
            facecolor=BASE.SCENE_COLORS[scene],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.88,
            zorder=3,
        )
    ax.plot([0.7, 6.3], [0.7, 6.3], color="#666666", linestyle=":", linewidth=0.8)
    ax.text(
        0.04,
        0.96,
        "30/45 configuration pairs\ndiscordant",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        color="#333333",
    )
    ax.set_title("(b)  Post hoc rank comparison", loc="left", fontweight="bold")
    ax.set_xlabel("Nominal macro-F1 rank (1 = best)")
    ax.set_ylabel("Mean-loss rank (1 = best)")
    ax.set_xticks(range(1, 7))
    ax.set_yticks(range(1, 7))
    ax.set_xlim(0.65, 6.35)
    ax.set_ylim(6.35, 0.65)
    ax.grid(color="#E0E0E0", linewidth=0.4, alpha=0.75)
    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=0.6, pad=1.8)


def build_figure(
    dose_rows: list[dict[str, str]],
    nominal_reliability_rows: list[dict[str, str]],
    primary_rows: list[dict[str, str]],
    geometry_rows: list[dict[str, str]],
    probabilistic_rows: list[dict[str, str]],
) -> mpl.figure.Figure:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8.6,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.16, 4.78), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        6,
        left=0.064,
        right=0.992,
        bottom=0.095,
        top=0.975,
        wspace=1.00,
        hspace=0.42,
    )
    ax_a = fig.add_subplot(outer[0, 0:2])
    ax_b = fig.add_subplot(outer[0, 2:4])
    ax_c = fig.add_subplot(outer[0, 4:6])
    ax_d = fig.add_subplot(outer[1, 0:3])

    BASE.panel_dose(ax_a, dose_rows)
    panel_nominal_reliability(ax_b, nominal_reliability_rows)
    BASE.panel_transitions(ax_c, primary_rows)
    ax_c.set_title("(c)  Original-response transitions", loc="left", fontweight="bold")
    BASE.panel_geometry(ax_d, geometry_rows)
    ax_d.set_title("(d)  Virtual-response sensitivity", loc="left", fontweight="bold")

    inner = outer[1, 3:6].subgridspec(1, 3, wspace=0.60)
    calibration_axes = [fig.add_subplot(inner[0, index]) for index in range(3)]
    BASE.calibration_strip(
        calibration_axes[0], probabilistic_rows, "nll_change", r"$\Delta$NLL", 1.0, 36
    )
    BASE.calibration_strip(
        calibration_axes[1], probabilistic_rows, "brier_change", r"$\Delta$Brier (pp)", 100.0, 35
    )
    BASE.calibration_strip(
        calibration_axes[2], probabilistic_rows, "ece_change", r"$\Delta$ECE (pp)", 100.0, 31
    )
    calibration_axes[0].text(
        -0.30,
        1.13,
        "(e)  Original-response probabilistic metrics",
        transform=calibration_axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=BASE.SCENE_COLORS[scene],
            markeredgecolor="none",
            markersize=4.2,
            label=scene,
        )
        for scene in BASE.SCENES
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="#555555",
                linestyle="none",
                markersize=4.0,
                label=r"$-2.2$ nm",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="#555555",
                linestyle="none",
                markersize=4.0,
                label=r"$+2.2$ nm",
            ),
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
    parser.add_argument(
        "--dose", type=Path, default=ROOT / "data/reference_panel/dose_monotonicity.csv"
    )
    parser.add_argument(
        "--nominal-reliability",
        type=Path,
        default=ROOT / "data/reference_secondary/nominal_reliability_cells.csv",
    )
    parser.add_argument(
        "--primary",
        type=Path,
        default=ROOT / "data/reference_panel/primary_contrasts.csv",
    )
    parser.add_argument(
        "--geometry",
        type=Path,
        default=ROOT / "data/geometry_panel/geometry_primary_results.csv",
    )
    parser.add_argument(
        "--probabilistic",
        type=Path,
        default=ROOT / "data/reference_secondary/primary_probabilistic_metrics.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "assets/images/Fig2_sensitivity_audit_v18.pdf",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figure = build_figure(
        BASE.read_csv(args.dose),
        BASE.read_csv(args.nominal_reliability),
        BASE.read_csv(args.primary),
        BASE.read_csv(args.geometry),
        BASE.read_csv(args.probabilistic),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(
        args.output.with_suffix(".png"),
        dpi=500,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
