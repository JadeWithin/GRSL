#!/usr/bin/env python3
"""Verify the manuscript's reported audit numbers from released derived outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DIRECTION_CONDITIONS = {
    "minus": "global_center_minus_2p2_nm",
    "plus": "global_center_plus_2p2_nm",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual: float, expected: float, atol: float = 5e-3) -> None:
    if not np.isclose(actual, expected, atol=atol, rtol=0.0):
        raise AssertionError(f"expected {expected}, got {actual}")


def quantile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), 100.0 * q))


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_sign_counts(per_seed_root: Path) -> dict[tuple[str, str, str], int]:
    counts: dict[tuple[str, str, str], int] = {}
    for path in sorted(per_seed_root.glob("*/*.csv")):
        scene = path.parent.name
        model_key = path.stem
        rows = read_csv(path)
        values = {
            (row["seed"], row["condition"]): float(row["macro_f1"])
            for row in rows
        }
        seeds = sorted({row["seed"] for row in rows}, key=int)
        if len(seeds) != 5:
            raise AssertionError(f"{path}: expected five seeds, got {len(seeds)}")
        for direction, condition in DIRECTION_CONDITIONS.items():
            count = sum(
                values[(seed, "nominal")] - values[(seed, condition)] > 0.0
                for seed in seeds
            )
            counts[(scene, model_key, direction)] = count
    if len(counts) != 36:
        raise AssertionError(f"expected 36 seed-sign contrasts, got {len(counts)}")
    return counts


def summarize_proxy(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["geometry"], row["direction"])].append(row)
    summaries: list[dict[str, object]] = []
    order = ("dense_narrow", "reference", "coarse_broad")
    for geometry in order:
        for direction in ("minus", "plus"):
            subset = grouped[(geometry, direction)]
            if len(subset) != 30:
                raise AssertionError(
                    f"{geometry}/{direction}: expected 30 proxy spectra, got {len(subset)}"
                )
            q = [float(row["perturbation_error_ratio"]) for row in subset]
            nrmse = [float(row["shifted_relative_nrmse"]) for row in subset]
            angle = [float(row["shifted_spectral_angle_deg"]) for row in subset]
            first = subset[0]
            summaries.append(
                {
                    "geometry": geometry,
                    "band_count": int(first["band_count"]),
                    "spacing_nm": float(first["spacing_nm"]),
                    "effective_fwhm_nm": float(first["effective_fwhm_nm"]),
                    "direction": direction,
                    "shift_nm": float(first["shift_nm"]),
                    "spectrum_count": len(subset),
                    "perturbation_error_ratio_median": float(np.median(q)),
                    "perturbation_error_ratio_p95": float(np.percentile(q, 95)),
                    "perturbation_error_ratio_max": float(np.max(q)),
                    "shifted_relative_nrmse_median": float(np.median(nrmse)),
                    "shifted_spectral_angle_deg_median": float(np.median(angle)),
                }
            )
    return summaries


def write_proxy_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/audit_report.json")
    args = parser.parse_args()

    data = ROOT / "data"
    analysis = ROOT / "outputs" / "analysis"

    primary = read_csv(data / "reference_panel" / "primary_contrasts.csv")
    dose = read_csv(data / "reference_panel" / "dose_monotonicity.csv")
    all_conditions = read_csv(data / "reference_panel" / "all_conditions.csv")
    probabilistic = read_csv(
        data / "reference_secondary" / "primary_probabilistic_metrics.csv"
    )
    blocks = read_csv(data / "reference_secondary" / "block_size_sensitivity.csv")
    geometry = read_csv(data / "geometry_panel" / "geometry_primary_results.csv")
    seed_variability = read_csv(
        data / "reproducibility" / "geometry_five_seed_variability.csv"
    )
    parameters = read_csv(
        data / "reproducibility" / "geometry_parameter_counts.csv"
    )
    proxy_rows = read_csv(
        data / "proxy_validation" / "geometry_proxy_per_spectrum.csv"
    )

    if len(primary) != 36:
        raise AssertionError(f"expected 36 primary contrasts, got {len(primary)}")
    primary_drops = [float(row["mean_drop_pp"]) for row in primary]
    if sum(value > 0.0 for value in primary_drops) != 36:
        raise AssertionError("not all primary five-seed mean drops are positive")
    close(min(primary_drops), 0.93, 0.01)
    close(max(primary_drops), 25.20, 0.01)

    sign_counts = seed_sign_counts(data / "reference_panel" / "per_seed")
    if sum(count == 5 for count in sign_counts.values()) != 35:
        raise AssertionError("expected 35/36 contrasts positive in all five seeds")

    bootstrap = np.load(
        data / "reference_panel" / "primary_block_bootstrap.npz",
        allow_pickle=False,
    )
    ci_positive = 0
    for row in primary:
        key = f'{row["scene"]}__{row["model_key"]}__{row["direction"]}'
        samples = np.asarray(bootstrap[key], dtype=np.float64)
        if samples.shape != (2000,):
            raise AssertionError(f"{key}: unexpected bootstrap shape {samples.shape}")
        lower, upper = np.percentile(samples, [2.5, 97.5])
        close(lower, float(row["block_bootstrap_ci95_lower_pp"]), 2e-5)
        close(upper, float(row["block_bootstrap_ci95_upper_pp"]), 2e-5)
        ci_positive += int(lower > 0.0)
    if ci_positive != 33:
        raise AssertionError(f"expected 33/36 positive primary block CIs, got {ci_positive}")

    if sum(row["fully_nondecreasing"].lower() == "true" for row in dose) != 34:
        raise AssertionError("expected 34/36 fully nondecreasing dose curves")

    block_groups: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    for row in blocks:
        block_groups[(row["scene"], row["model_key"], row["direction"])].append(
            row["ci_excludes_zero"].lower() == "true"
        )
    stable = sum(len(values) == 3 and all(values) for values in block_groups.values())
    if len(block_groups) != 36 or stable != 33:
        raise AssertionError("alternative block-scale stability is not 33/36")

    nll = [float(row["nll_change"]) for row in probabilistic]
    brier = [float(row["brier_change"]) for row in probabilistic]
    ece = [float(row["ece_change"]) for row in probabilistic]
    if (sum(v > 0 for v in nll), sum(v > 0 for v in brier), sum(v > 0 for v in ece)) != (36, 35, 31):
        raise AssertionError("probabilistic endpoint counts differ from 36/35/31")
    close(100.0 * float(np.median(ece)), 3.37, 0.01)
    close(100.0 * max(ece), 19.20, 0.01)
    close(100.0 * max(float(row["correct_to_wrong_rate_mean"]) for row in primary), 28.30, 0.01)
    close(100.0 * max(float(row["label_flip_rate_mean"]) for row in primary), 36.28, 0.01)

    wide: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in all_conditions:
        wide[(row["scene"], row["model_key"])][row["condition"]] = float(
            row["macro_f1_mean"]
        )

    def control(condition: str) -> tuple[int, float]:
        drops = [100.0 * (cell["nominal"] - cell[condition]) for cell in wide.values()]
        return sum(value > 0.0 for value in drops), float(np.median(drops))

    controls = {
        "local_minus": control("local_mid25_center_minus_2p2_nm"),
        "local_plus": control("local_mid25_center_plus_2p2_nm"),
        "fwhm_12": control("fwhm_12_nm"),
        "fwhm_18": control("fwhm_18_nm"),
    }
    expected_controls = {
        "local_minus": (17, 3.37),
        "local_plus": (17, 3.58),
        "fwhm_12": (18, 0.51),
        "fwhm_18": (15, 0.42),
    }
    for name, (count, median) in expected_controls.items():
        if controls[name][0] != count:
            raise AssertionError(f"{name}: expected {count} positive cells")
        close(controls[name][1], median, 0.01)

    if len(geometry) != 108:
        raise AssertionError(f"expected 108 geometry contrasts, got {len(geometry)}")
    if sum(float(row["paired_macro_f1_drop_mean"]) > 0.0 for row in geometry) != 108:
        raise AssertionError("not all geometry contrasts have positive mean loss")

    geometry_stats: dict[str, dict[str, float | int]] = {}
    geometry_expected = {
        "dense_narrow": (9.51, 5.95, 14.22, 1.69),
        "reference": (7.54, 5.18, 11.99, 1.27),
        "coarse_broad": (7.91, 4.42, 11.89, 1.24),
    }
    ratio_expected = {
        "dense_narrow": (0.365, 0.244),
        "reference": (0.275, 0.147),
        "coarse_broad": (0.185, 0.122),
    }
    sign_groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in geometry:
        sign_groups[(row["scene"], row["model_key"], row["direction"])].append(
            float(row["paired_macro_f1_drop_mean"])
        )
    if len(sign_groups) != 36 or not all(
        len(values) == 3 and all(value > 0.0 for value in values)
        for values in sign_groups.values()
    ):
        raise AssertionError("geometry sign consistency is not 36/36")

    for name, expected in geometry_expected.items():
        rows = [row for row in geometry if row["geometry"] == name]
        drops = [100.0 * float(row["paired_macro_f1_drop_mean"]) for row in rows]
        sds = [
            float(row["five_seed_drop_sd_pp"])
            for row in seed_variability
            if row["geometry"] == name
        ]
        actual = (
            float(np.median(drops)),
            quantile(drops, 0.25),
            quantile(drops, 0.75),
            float(np.median(sds)),
        )
        for value, target in zip(actual, expected):
            close(value, target, 0.01)
        first = rows[0]
        close(float(first["absolute_shift_over_spacing"]), ratio_expected[name][0], 5e-4)
        close(float(first["absolute_shift_over_fwhm"]), ratio_expected[name][1], 5e-4)
        geometry_stats[name] = {
            "positive": len(rows),
            "median_pp": actual[0],
            "q1_pp": actual[1],
            "q3_pp": actual[2],
            "median_seed_sd_pp": actual[3],
        }

    parameter_values = [int(row["trainable_parameters"]) for row in parameters]
    if len(parameters) != 54 or (min(parameter_values), max(parameter_values)) != (423, 148197):
        raise AssertionError("parameter-count range differs from 423--148197")

    proxy = summarize_proxy(proxy_rows)
    write_proxy_summary(ROOT / "outputs" / "proxy_summary.csv", proxy)
    proxy_expected = {
        ("dense_narrow", "minus"): 1.41e-5,
        ("dense_narrow", "plus"): 1.46e-5,
        ("reference", "minus"): 2.63e-5,
        ("reference", "plus"): 6.21e-6,
        ("coarse_broad", "minus"): 8.38e-4,
        ("coarse_broad", "plus"): 2.83e-4,
    }
    for row in proxy:
        expected = proxy_expected[(str(row["geometry"]), str(row["direction"]))]
        close(
            float(row["perturbation_error_ratio_p95"]),
            expected,
            max(5e-8, abs(expected) * 0.01),
        )

    nominal = load_json(analysis / "nominal_reliability_summary.json")
    relative = load_json(analysis / "nominal_reliability_relative_summary.json")
    policy = load_json(analysis / "nominal_reliability_loss_policy_summary.json")
    if (
        nominal["within_scene_pair_count"],
        nominal["mean_loss_rank_discordant_count"],
        nominal["worst_loss_rank_discordant_count"],
        nominal["top_nominal_differs_from_lowest_mean_loss_scene_count"],
    ) != (45, 30, 31, 3):
        raise AssertionError("absolute-loss ordering summary differs from 45/30/31/3")
    if (
        relative["within_scene_pair_count"],
        relative["mean_relative_loss_rank_discordant_count"],
        relative["worst_relative_loss_rank_discordant_count"],
        relative["top_nominal_differs_from_lowest_mean_relative_loss_scene_count"],
    ) != (45, 30, 27, 3):
        raise AssertionError("relative-loss ordering summary differs from 45/30/27/3")
    within_policy = policy["within_loss_policy_pairs"]
    if (
        within_policy["pair_count"],
        within_policy["mean_loss_rank_discordant_count"],
        within_policy["worst_loss_rank_discordant_count"],
    ) != (18, 13, 14):
        raise AssertionError("shared-loss-policy summary differs from 18/13/14")

    required_outputs = [
        ROOT / "outputs" / "Table_I.tex",
        ROOT / "outputs" / "Fig2_sensitivity_audit.pdf",
        ROOT / "outputs" / "Fig2_sensitivity_audit.png",
    ]
    missing = [str(path) for path in required_outputs if not path.is_file()]
    if missing:
        raise AssertionError(f"missing generated outputs: {missing}")

    report = {
        "status": "PASS",
        "scope": "analysis-level reproduction from released fixed derived outputs",
        "primary_original_response": {
            "positive_five_seed_mean": "36/36",
            "positive_in_all_five_seeds": "35/36",
            "mean_loss_range_pp": [round(min(primary_drops), 2), round(max(primary_drops), 2)],
            "primary_block_intervals_above_zero": "33/36",
            "dose_nondecreasing": "34/36",
        },
        "ordering": {
            "within_scene_pairs": 45,
            "absolute_mean_loss_discordant": "30/45",
            "absolute_worst_loss_discordant": "31/45",
            "relative_mean_loss_discordant": "30/45",
            "relative_worst_loss_discordant": "27/45",
            "shared_loss_policy_mean_loss_discordant": "13/18",
        },
        "probabilistic": {
            "nll_increase": "36/36",
            "brier_increase": "35/36",
            "ece_increase": "31/36",
        },
        "geometry": geometry_stats,
        "parameter_range": [423, 148197],
        "proxy_geometry_direction_rows": 6,
        "generated_outputs": [path.relative_to(ROOT).as_posix() for path in required_outputs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
