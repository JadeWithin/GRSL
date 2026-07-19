#!/usr/bin/env python3
"""Generate Table I with seed-sign consistency and spatial-block intervals."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

SCENES = ("LongKou", "HanChuan", "HongHu")
MODELS = (
    ("Linear", "linear", "Linear"),
    ("MLP", "mlp", "MLP"),
    ("CNN-1D", "cnn1d", "CNN-1D"),
    ("SSCNN-3x3", "sscnn3x3", r"SSCNN-$3\times3$"),
    ("SSResNet-3x3", "ssresnet3x3", r"SSResNet-$3\times3$"),
    ("SpectralFormer-3x3", "spectralformer3x3", r"SpectralFormer-$3\times3$"),
)
DIRECTIONS = ("minus", "plus")
CONDITIONS = {
    "minus": "global_center_minus_2p2_nm",
    "plus": "global_center_plus_2p2_nm",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_panel(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    rows = read_csv(path)
    indexed = {(row["scene"], row["model"], row["direction"]): row for row in rows}
    required = {
        (scene, model, direction)
        for scene in SCENES
        for model, _, _ in MODELS
        for direction in DIRECTIONS
    }
    if set(indexed) != required:
        raise ValueError(
            f"Frozen panel mismatch: missing={sorted(required - set(indexed))}, "
            f"extra={sorted(set(indexed) - required)}"
        )
    return indexed


def seed_signs(per_seed_root: Path) -> dict[tuple[str, str, str], tuple[int, int]]:
    output: dict[tuple[str, str, str], tuple[int, int]] = {}
    for scene in SCENES:
        for model, model_key, _ in MODELS:
            rows = read_csv(per_seed_root / scene / f"{model_key}.csv")
            values = {
                (row["seed"], row["condition"]): float(row["macro_f1"])
                for row in rows
            }
            seeds = sorted({row["seed"] for row in rows}, key=int)
            for direction in DIRECTIONS:
                drops = [
                    values[(seed, "nominal")]
                    - values[(seed, CONDITIONS[direction])]
                    for seed in seeds
                ]
                output[(scene, model, direction)] = (
                    sum(drop > 0.0 for drop in drops),
                    len(drops),
                )
    return output


def contrast(row: dict[str, str]) -> str:
    return (
        f'{float(row["mean_drop_pp"]):.2f} '
        f'[{float(row["block_bootstrap_ci95_lower_pp"]):.2f}, '
        f'{float(row["block_bootstrap_ci95_upper_pp"]):.2f}]'
    )


def generate(input_csv: Path, per_seed_root: Path, output_tex: Path) -> None:
    rows = read_panel(input_csv)
    signs = seed_signs(per_seed_root)
    br = r"\\"
    lines = [
        "% Generated from the authoritative primary and per-seed CSV files.",
        r"\begin{table*}[!t]",
        r"\caption{Post Hoc Complete-Panel Results Under the Original Virtual Response at the Main $\pm2.2$-nm Center Shift (68 Bands, 8-nm Spacing, 15-nm FWHM)}",
        r"\label{tab:strategy_a_primary}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{1.8pt}",
        r"\renewcommand{\arraystretch}{0.98}",
        r"\begin{tabular}{llcccc}",
        r"\hline",
        "Scene & Configuration & Nominal F1 & $\\Delta$F1 at $-2.2$ nm [CI] & "
        "$\\Delta$F1 at $+2.2$ nm [CI] & Positive seeds $-/+$ " + br,
        r"\hline",
    ]
    for scene_index, scene in enumerate(SCENES):
        for model, _, model_tex in MODELS:
            minus = rows[(scene, model, "minus")]
            plus = rows[(scene, model, "plus")]
            nominal = float(minus["nominal_macro_f1"])
            if abs(nominal - float(plus["nominal_macro_f1"])) > 1e-12:
                raise ValueError(f"Nominal mismatch: {scene}/{model}")
            minus_sign = signs[(scene, model, "minus")]
            plus_sign = signs[(scene, model, "plus")]
            sign_text = (
                f"{minus_sign[0]}/{minus_sign[1]};"
                f"{plus_sign[0]}/{plus_sign[1]}"
            )
            lines.append(
                f"{scene} & {model_tex} & {nominal:.4f} & {contrast(minus)} & "
                f"{contrast(plus)} & {sign_text} {br}"
            )
        if scene_index < len(SCENES) - 1:
            lines.append(r"\hline")
    lines.extend(
        [
            r"\hline",
            r"\multicolumn{6}{l}{\footnotesize $\Delta$F1 is the five-seed mean nominal-minus-shifted macro-F1 in percentage points.} " + br,
            r"\multicolumn{6}{l}{\footnotesize Brackets are descriptive, conditional, unadjusted percentile 95\% intervals from 2000 paired spatial-block bootstrap replicates.} " + br,
            r"\multicolumn{6}{l}{\footnotesize Positive seeds count $\Delta$F1 above zero among five paired training seeds, reported as minus/plus.} " + br,
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--per-seed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.input, args.per_seed_root, args.output)


if __name__ == "__main__":
    main()
