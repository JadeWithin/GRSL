#!/usr/bin/env python3
"""Check whether nominal--reliability rank discordance survives loss normalization.

This post hoc sensitivity analysis divides each absolute macro-F1 drop by the
cell's nominal macro-F1. It addresses the bounded-scale concern that a model
with higher nominal macro-F1 has more available absolute loss. The primary
percentage-point analysis remains unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path


SCENES = ("LongKou", "HanChuan", "HongHu")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells",
        type=Path,
        default=root / "data/reference_secondary/nominal_reliability_cells.csv",
    )
    parser.add_argument(
        "--absolute-summary",
        type=Path,
        default=root / "data/reference_secondary/nominal_reliability_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data/reference_secondary",
    )
    parser.add_argument("--manuscript", type=Path, default=root / "main.tex")
    parser.add_argument(
        "--qc-output",
        type=Path,
        default=root / "qc/nominal_reliability_audit_v18.json",
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


def ordinal_ranks(values: list[float]) -> list[int]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0] * len(values)
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank
    return ranks


def build_relative_cells(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    if len(rows) != 18:
        raise ValueError(f"expected 18 scene--model cells, got {len(rows)}")
    output: list[dict[str, object]] = []
    for scene in SCENES:
        scene_rows = [row for row in rows if row["scene"] == scene]
        if len(scene_rows) != 6:
            raise ValueError(f"expected six models for {scene}, got {len(scene_rows)}")
        prepared: list[dict[str, object]] = []
        for row in scene_rows:
            nominal = float(row["nominal_macro_f1"])
            if not 0.0 < nominal <= 1.0:
                raise ValueError(f"invalid nominal macro-F1 for {scene}/{row['model']}: {nominal}")
            prepared.append(
                {
                    "scene": scene,
                    "model": row["model"],
                    "model_key": row["model_key"],
                    "nominal_macro_f1": nominal,
                    "mean_directional_drop_pp": float(row["mean_directional_drop_pp"]),
                    "worst_direction_drop_pp": float(row["worst_direction_drop_pp"]),
                    "mean_relative_loss": float(row["mean_directional_drop_pp"])
                    / (100.0 * nominal),
                    "worst_direction_relative_loss": float(row["worst_direction_drop_pp"])
                    / (100.0 * nominal),
                    "analysis_role": "post_hoc_bounded_scale_sensitivity",
                }
            )
        mean_ranks = ordinal_ranks([float(row["mean_relative_loss"]) for row in prepared])
        worst_ranks = ordinal_ranks(
            [float(row["worst_direction_relative_loss"]) for row in prepared]
        )
        for row, mean_rank, worst_rank in zip(prepared, mean_ranks, worst_ranks):
            row["mean_relative_loss_rank"] = mean_rank
            row["worst_relative_loss_rank"] = worst_rank
        output.extend(prepared)
    return output


def build_pairs(cells: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for scene in SCENES:
        scene_cells = [row for row in cells if row["scene"] == scene]
        for left, right in combinations(scene_cells, 2):
            nominal_delta = float(left["nominal_macro_f1"]) - float(
                right["nominal_macro_f1"]
            )
            mean_delta = float(left["mean_relative_loss"]) - float(
                right["mean_relative_loss"]
            )
            worst_delta = float(left["worst_direction_relative_loss"]) - float(
                right["worst_direction_relative_loss"]
            )
            higher_nominal = left if nominal_delta > 0 else right
            lower_mean = left if mean_delta < 0 else right
            lower_worst = left if worst_delta < 0 else right
            output.append(
                {
                    "scene": scene,
                    "model_a": left["model"],
                    "model_b": right["model"],
                    "higher_nominal_model": higher_nominal["model"],
                    "lower_mean_relative_loss_model": lower_mean["model"],
                    "lower_worst_relative_loss_model": lower_worst["model"],
                    "mean_relative_loss_rank_discordant": nominal_delta * mean_delta > 0,
                    "worst_relative_loss_rank_discordant": nominal_delta * worst_delta > 0,
                    "absolute_nominal_gap_pp": 100.0 * abs(nominal_delta),
                    "absolute_mean_relative_loss_gap": abs(mean_delta),
                    "absolute_worst_relative_loss_gap": abs(worst_delta),
                    "analysis_role": "post_hoc_bounded_scale_sensitivity",
                }
            )
    if len(output) != 45:
        raise ValueError(f"expected 45 within-scene pairs, got {len(output)}")
    return output


def summarize(
    cells: list[dict[str, object]], pairs: list[dict[str, object]]
) -> dict[str, object]:
    per_scene: dict[str, dict[str, object]] = {}
    for scene in SCENES:
        scene_cells = [row for row in cells if row["scene"] == scene]
        scene_pairs = [row for row in pairs if row["scene"] == scene]
        top_nominal = max(scene_cells, key=lambda row: float(row["nominal_macro_f1"]))
        lowest_mean = min(scene_cells, key=lambda row: float(row["mean_relative_loss"]))
        lowest_worst = min(
            scene_cells, key=lambda row: float(row["worst_direction_relative_loss"])
        )
        per_scene[scene] = {
            "pair_count": len(scene_pairs),
            "mean_relative_loss_rank_discordant_count": sum(
                bool(row["mean_relative_loss_rank_discordant"]) for row in scene_pairs
            ),
            "worst_relative_loss_rank_discordant_count": sum(
                bool(row["worst_relative_loss_rank_discordant"]) for row in scene_pairs
            ),
            "top_nominal_model": top_nominal["model"],
            "lowest_mean_relative_loss_model": lowest_mean["model"],
            "lowest_worst_relative_loss_model": lowest_worst["model"],
            "top_nominal_differs_from_lowest_mean_relative_loss": top_nominal["model"]
            != lowest_mean["model"],
            "top_nominal_differs_from_lowest_worst_relative_loss": top_nominal["model"]
            != lowest_worst["model"],
        }

    mean_count = sum(bool(row["mean_relative_loss_rank_discordant"]) for row in pairs)
    worst_count = sum(bool(row["worst_relative_loss_rank_discordant"]) for row in pairs)
    return {
        "analysis_role": "post_hoc_bounded_scale_sensitivity",
        "definition": "absolute_macro_f1_drop_divided_by_nominal_macro_f1",
        "primary_analysis_unchanged": True,
        "cell_count": len(cells),
        "within_scene_pair_count": len(pairs),
        "mean_relative_loss_rank_discordant_count": mean_count,
        "mean_relative_loss_rank_discordant_fraction": mean_count / len(pairs),
        "worst_relative_loss_rank_discordant_count": worst_count,
        "worst_relative_loss_rank_discordant_fraction": worst_count / len(pairs),
        "top_nominal_differs_from_lowest_mean_relative_loss_scene_count": sum(
            bool(values["top_nominal_differs_from_lowest_mean_relative_loss"])
            for values in per_scene.values()
        ),
        "top_nominal_differs_from_lowest_worst_relative_loss_scene_count": sum(
            bool(values["top_nominal_differs_from_lowest_worst_relative_loss"])
            for values in per_scene.values()
        ),
        "per_scene": per_scene,
    }


def build_qc(
    absolute: dict[str, object],
    relative: dict[str, object],
    manuscript_path: Path,
) -> dict[str, object]:
    manuscript = manuscript_path.read_text(encoding="utf-8")
    required_tokens = [
        "conditional finite-test-set audit estimand",
        "A relative-loss sensitivity divided each loss by the cell's nominal macro-F1",
        "Dividing losses by nominal macro-F1 left the mean-loss discordance count unchanged",
        "a post hoc descriptive analysis further shows frequent divergence",
        "domain-generalization augmentation \\cite{Chen2026SPDDA}",
        "proposed data-level correction for intensity and transversal shifts",
    ]
    forbidden_tokens = [
        "They do not isolate a downstream qualification question",
        "Chen2026SPDDA,Musiat2026SSFT",
        "Vaquet \\textit{et al.} established degradation under wavelength-axis shifts",
    ]
    missing = [token for token in required_tokens if token not in manuscript]
    forbidden_present = [token for token in forbidden_tokens if token in manuscript]
    expected = {
        "absolute_mean": 30,
        "absolute_worst": 31,
        "relative_mean": 30,
        "relative_worst": 27,
        "relative_top_mismatch": 3,
    }
    observed = {
        "absolute_mean": int(absolute["mean_loss_rank_discordant_count"]),
        "absolute_worst": int(absolute["worst_loss_rank_discordant_count"]),
        "relative_mean": int(relative["mean_relative_loss_rank_discordant_count"]),
        "relative_worst": int(relative["worst_relative_loss_rank_discordant_count"]),
        "relative_top_mismatch": int(
            relative["top_nominal_differs_from_lowest_mean_relative_loss_scene_count"]
        ),
    }
    numeric_checks = {key: observed[key] == value for key, value in expected.items()}
    return {
        "status": "pass"
        if all(numeric_checks.values()) and not missing and not forbidden_present
        else "fail",
        "analysis_role": "post_hoc_descriptive_with_bounded_scale_sensitivity",
        "primary_analysis_unchanged": True,
        "primary_contrast_count": 36,
        "cell_count": int(absolute["cell_count"]),
        "within_scene_pair_count": int(absolute["within_scene_pair_count"]),
        "mean_loss_rank_discordant_count": observed["absolute_mean"],
        "worst_loss_rank_discordant_count": observed["absolute_worst"],
        "mean_relative_loss_rank_discordant_count": observed["relative_mean"],
        "worst_relative_loss_rank_discordant_count": observed["relative_worst"],
        "top_nominal_differs_from_lowest_mean_loss_scene_count": int(
            absolute["top_nominal_differs_from_lowest_mean_loss_scene_count"]
        ),
        "top_nominal_differs_from_lowest_mean_relative_loss_scene_count": observed[
            "relative_top_mismatch"
        ],
        "relative_mean_count_matches_absolute_mean_count": observed["relative_mean"]
        == observed["absolute_mean"],
        "numeric_checks": numeric_checks,
        "missing_manuscript_tokens": missing,
        "forbidden_manuscript_tokens_present": forbidden_present,
    }


def main() -> None:
    args = parse_args()
    rows = read_csv(args.cells)
    cells = build_relative_cells(rows)
    pairs = build_pairs(cells)
    summary = summarize(cells, pairs)
    absolute = json.loads(args.absolute_summary.read_text(encoding="utf-8"))
    qc = build_qc(absolute, summary, args.manuscript)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "nominal_reliability_relative_cells.csv", cells)
    write_csv(args.output_dir / "nominal_reliability_relative_pairwise.csv", pairs)
    (args.output_dir / "nominal_reliability_relative_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.qc_output.parent.mkdir(parents=True, exist_ok=True)
    args.qc_output.write_text(
        json.dumps(qc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"relative": summary, "qc": qc}, ensure_ascii=False, indent=2))
    if qc["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()