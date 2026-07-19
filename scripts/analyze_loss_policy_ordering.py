#!/usr/bin/env python3
"""Audit nominal--shift ordering within shared loss-weighting policies.

This post hoc descriptive check uses the already frozen 45 within-scene
configuration pairs. It does not retrain, select, or exclude any configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LOSS_POLICY = {
    "Linear": "unweighted_cross_entropy",
    "MLP": "unweighted_cross_entropy",
    "CNN-1D": "unweighted_cross_entropy",
    "SSCNN-3x3": "inverse_frequency_weighted_cross_entropy",
    "SSResNet-3x3": "inverse_frequency_weighted_cross_entropy",
    "SpectralFormer-3x3": "inverse_frequency_weighted_cross_entropy",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=root / "data/reference_secondary/nominal_reliability_pairwise.csv",
    )
    parser.add_argument(
        "--annotated-output",
        type=Path,
        default=root
        / "data/reference_secondary/nominal_reliability_loss_policy_pairwise.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=root
        / "data/reference_secondary/nominal_reliability_loss_policy_summary.json",
    )
    parser.add_argument(
        "--qc-output",
        type=Path,
        default=root / "qc/nominal_reliability_loss_policy_audit_v18.json",
    )
    return parser.parse_args()


def as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"invalid Boolean value: {value!r}")


def main() -> None:
    args = parse_args()
    with args.pairs.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 45:
        raise ValueError(f"expected 45 within-scene pairs, got {len(rows)}")

    annotated: list[dict[str, object]] = []
    for row in rows:
        policy_a = LOSS_POLICY[row["model_a"]]
        policy_b = LOSS_POLICY[row["model_b"]]
        same_policy = policy_a == policy_b
        annotated.append(
            {
                **row,
                "model_a_loss_policy": policy_a,
                "model_b_loss_policy": policy_b,
                "same_loss_weighting_policy": same_policy,
                "comparison_group": "within_policy" if same_policy else "cross_policy",
                "analysis_role": "post_hoc_loss_policy_sensitivity",
            }
        )

    def summarize(subset: list[dict[str, object]]) -> dict[str, object]:
        return {
            "pair_count": len(subset),
            "mean_loss_rank_discordant_count": sum(
                as_bool(str(row["mean_loss_rank_discordant"])) for row in subset
            ),
            "worst_loss_rank_discordant_count": sum(
                as_bool(str(row["worst_loss_rank_discordant"])) for row in subset
            ),
        }

    within = [row for row in annotated if bool(row["same_loss_weighting_policy"])]
    cross = [row for row in annotated if not bool(row["same_loss_weighting_policy"])]
    summary = {
        "analysis_role": "post_hoc_loss_policy_sensitivity",
        "primary_analysis_unchanged": True,
        "audited_unit": (
            "complete trained configuration including architecture, loss weighting, "
            "normalization, and validation-selected checkpoint"
        ),
        "interpretation": (
            "Persistence within a shared loss-weighting policy reduces, but does not "
            "eliminate, confounding; this is not a causal architecture comparison."
        ),
        "all_pairs": summarize(annotated),
        "within_loss_policy_pairs": summarize(within),
        "cross_loss_policy_pairs": summarize(cross),
    }
    expected = {
        "all_pairs": {
            "pair_count": 45,
            "mean_loss_rank_discordant_count": 30,
            "worst_loss_rank_discordant_count": 31,
        },
        "within_loss_policy_pairs": {
            "pair_count": 18,
            "mean_loss_rank_discordant_count": 13,
            "worst_loss_rank_discordant_count": 14,
        },
        "cross_loss_policy_pairs": {
            "pair_count": 27,
            "mean_loss_rank_discordant_count": 17,
            "worst_loss_rank_discordant_count": 17,
        },
    }
    checks = {
        group: {
            key: int(summary[group][key]) == value
            for key, value in expected[group].items()
        }
        for group in expected
    }
    qc = {
        "status": "pass"
        if all(value for group in checks.values() for value in group.values())
        else "fail",
        "checks": checks,
        "summary": summary,
    }

    args.annotated_output.parent.mkdir(parents=True, exist_ok=True)
    with args.annotated_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(annotated[0]))
        writer.writeheader()
        writer.writerows(annotated)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.qc_output.parent.mkdir(parents=True, exist_ok=True)
    args.qc_output.write_text(
        json.dumps(qc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(qc, ensure_ascii=False, indent=2))
    if qc["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()