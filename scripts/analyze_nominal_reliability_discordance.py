#!/usr/bin/env python3
"""Describe whether nominal accuracy and center-shift reliability rank models alike.

This is a post hoc, descriptive analysis of the frozen reference-response panel.
It does not retrain models, select cells, or alter the primary analysis.  Model
comparisons are made only within scenes because macro-F1 levels and class sets
differ across scenes.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np


SCENES = ("LongKou", "HanChuan", "HongHu")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary",
        type=Path,
        default=root / "data/reference_panel/primary_contrasts.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data/reference_secondary",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ordinal_ranks(values: list[float], *, reverse: bool) -> list[int]:
    """Return one-based ranks; the frozen panel has no ties for these values."""
    order = sorted(range(len(values)), key=values.__getitem__, reverse=reverse)
    ranks = [0] * len(values)
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    def average_ranks(values: list[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        order = np.argsort(array, kind="mergesort")
        ranks = np.empty(array.size, dtype=np.float64)
        start = 0
        while start < array.size:
            end = start + 1
            while end < array.size and array[order[end]] == array[order[start]]:
                end += 1
            ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
            start = end
        return ranks

    return float(np.corrcoef(average_ranks(a), average_ranks(b))[0, 1])


def build_cells(primary: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in primary:
        grouped.setdefault((row["scene"], row["model_key"]), []).append(row)

    cells: list[dict[str, object]] = []
    for scene in SCENES:
        scene_cells: list[dict[str, object]] = []
        for (row_scene, model_key), pair in grouped.items():
            if row_scene != scene:
                continue
            if len(pair) != 2 or {row["direction"] for row in pair} != {"minus", "plus"}:
                raise ValueError(f"expected two primary directions for {scene}/{model_key}")
            by_direction = {row["direction"]: row for row in pair}
            nominal_values = {float(row["nominal_macro_f1"]) for row in pair}
            if len(nominal_values) != 1:
                raise ValueError(f"nominal macro-F1 differs by direction for {scene}/{model_key}")
            minus_drop = float(by_direction["minus"]["mean_drop_pp"])
            plus_drop = float(by_direction["plus"]["mean_drop_pp"])
            scene_cells.append(
                {
                    "scene": scene,
                    "model": pair[0]["model"],
                    "model_key": model_key,
                    "nominal_macro_f1": nominal_values.pop(),
                    "minus_drop_pp": minus_drop,
                    "plus_drop_pp": plus_drop,
                    "mean_directional_drop_pp": 0.5 * (minus_drop + plus_drop),
                    "worst_direction_drop_pp": max(minus_drop, plus_drop),
                    "analysis_role": "post_hoc_descriptive",
                }
            )

        if len(scene_cells) != 6:
            raise ValueError(f"expected six models for {scene}, got {len(scene_cells)}")
        nominal_ranks = ordinal_ranks(
            [float(row["nominal_macro_f1"]) for row in scene_cells], reverse=True
        )
        mean_ranks = ordinal_ranks(
            [float(row["mean_directional_drop_pp"]) for row in scene_cells], reverse=False
        )
        worst_ranks = ordinal_ranks(
            [float(row["worst_direction_drop_pp"]) for row in scene_cells], reverse=False
        )
        for row, nominal_rank, mean_rank, worst_rank in zip(
            scene_cells, nominal_ranks, mean_ranks, worst_ranks
        ):
            row["nominal_rank"] = nominal_rank
            row["mean_loss_reliability_rank"] = mean_rank
            row["worst_loss_reliability_rank"] = worst_rank
        cells.extend(scene_cells)
    return cells


def build_pairs(cells: list[dict[str, object]]) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    for scene in SCENES:
        subset = [row for row in cells if row["scene"] == scene]
        for left, right in combinations(subset, 2):
            nominal_delta = float(left["nominal_macro_f1"]) - float(right["nominal_macro_f1"])
            mean_delta = float(left["mean_directional_drop_pp"]) - float(
                right["mean_directional_drop_pp"]
            )
            worst_delta = float(left["worst_direction_drop_pp"]) - float(
                right["worst_direction_drop_pp"]
            )
            higher_nominal = left if nominal_delta > 0 else right
            lower_mean = left if mean_delta < 0 else right
            lower_worst = left if worst_delta < 0 else right
            pairs.append(
                {
                    "scene": scene,
                    "model_a": left["model"],
                    "model_b": right["model"],
                    "higher_nominal_model": higher_nominal["model"],
                    "lower_mean_loss_model": lower_mean["model"],
                    "lower_worst_loss_model": lower_worst["model"],
                    "mean_loss_rank_discordant": nominal_delta * mean_delta > 0,
                    "worst_loss_rank_discordant": nominal_delta * worst_delta > 0,
                    "absolute_nominal_gap_pp": 100.0 * abs(nominal_delta),
                    "absolute_mean_loss_gap_pp": abs(mean_delta),
                    "absolute_worst_loss_gap_pp": abs(worst_delta),
                    "analysis_role": "post_hoc_descriptive",
                }
            )
    if len(pairs) != 45:
        raise ValueError(f"expected 45 within-scene pairs, got {len(pairs)}")
    return pairs


def summarize(
    cells: list[dict[str, object]], pairs: list[dict[str, object]]
) -> dict[str, object]:
    per_scene: dict[str, dict[str, object]] = {}
    for scene in SCENES:
        scene_cells = [row for row in cells if row["scene"] == scene]
        scene_pairs = [row for row in pairs if row["scene"] == scene]
        top_nominal = min(scene_cells, key=lambda row: int(row["nominal_rank"]))
        lowest_mean = min(
            scene_cells, key=lambda row: int(row["mean_loss_reliability_rank"])
        )
        lowest_worst = min(
            scene_cells, key=lambda row: int(row["worst_loss_reliability_rank"])
        )
        per_scene[scene] = {
            "pair_count": len(scene_pairs),
            "mean_loss_rank_discordant_count": sum(
                bool(row["mean_loss_rank_discordant"]) for row in scene_pairs
            ),
            "worst_loss_rank_discordant_count": sum(
                bool(row["worst_loss_rank_discordant"]) for row in scene_pairs
            ),
            "top_nominal_model": top_nominal["model"],
            "lowest_mean_loss_model": lowest_mean["model"],
            "lowest_worst_loss_model": lowest_worst["model"],
            "top_nominal_differs_from_lowest_mean_loss": top_nominal["model"]
            != lowest_mean["model"],
            "top_nominal_differs_from_lowest_worst_loss": top_nominal["model"]
            != lowest_worst["model"],
        }

    mean_discordant = sum(bool(row["mean_loss_rank_discordant"]) for row in pairs)
    worst_discordant = sum(bool(row["worst_loss_rank_discordant"]) for row in pairs)
    nominal = [float(row["nominal_macro_f1"]) for row in cells]
    mean_loss = [float(row["mean_directional_drop_pp"]) for row in cells]
    worst_loss = [float(row["worst_direction_drop_pp"]) for row in cells]
    within_nominal_ranks = [float(row["nominal_rank"]) for row in cells]
    within_mean_ranks = [float(row["mean_loss_reliability_rank"]) for row in cells]
    within_worst_ranks = [float(row["worst_loss_reliability_rank"]) for row in cells]
    return {
        "analysis_role": "post_hoc_descriptive",
        "primary_analysis_unchanged": True,
        "cell_count": len(cells),
        "within_scene_pair_count": len(pairs),
        "mean_loss_rank_discordant_count": mean_discordant,
        "mean_loss_rank_discordant_fraction": mean_discordant / len(pairs),
        "worst_loss_rank_discordant_count": worst_discordant,
        "worst_loss_rank_discordant_fraction": worst_discordant / len(pairs),
        "top_nominal_differs_from_lowest_mean_loss_scene_count": sum(
            bool(values["top_nominal_differs_from_lowest_mean_loss"])
            for values in per_scene.values()
        ),
        "top_nominal_differs_from_lowest_worst_loss_scene_count": sum(
            bool(values["top_nominal_differs_from_lowest_worst_loss"])
            for values in per_scene.values()
        ),
        "pooled_spearman_nominal_vs_mean_loss": spearman(nominal, mean_loss),
        "pooled_spearman_nominal_vs_worst_loss": spearman(nominal, worst_loss),
        "within_scene_rank_spearman_nominal_vs_mean_loss": spearman(
            within_nominal_ranks, within_mean_ranks
        ),
        "within_scene_rank_spearman_nominal_vs_worst_loss": spearman(
            within_nominal_ranks, within_worst_ranks
        ),
        "per_scene": per_scene,
    }


def main() -> None:
    args = parse_args()
    primary = read_csv(args.primary)
    if len(primary) != 36:
        raise ValueError(f"expected 36 primary contrasts, got {len(primary)}")
    cells = build_cells(primary)
    pairs = build_pairs(cells)
    summary = summarize(cells, pairs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "nominal_reliability_cells.csv", cells)
    write_csv(args.output_dir / "nominal_reliability_pairwise.csv", pairs)
    (args.output_dir / "nominal_reliability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
