#!/usr/bin/env python3
"""Run the public analysis-level reproduction capsule in one command."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OUTPUTS = ROOT / "outputs"
ANALYSIS = OUTPUTS / "analysis"
QC = OUTPUTS / "qc"


def run(*args: object) -> None:
    command = [sys.executable, *[str(value) for value in args]]
    print("+", " ".join(command), flush=True)
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail unless the generated audit report has status PASS.",
    )
    args = parser.parse_args()

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)

    run(
        SCRIPTS / "analyze_nominal_reliability_discordance.py",
        "--primary",
        ROOT / "data/reference_panel/primary_contrasts.csv",
        "--output-dir",
        ANALYSIS,
    )
    run(
        SCRIPTS / "analyze_relative_loss_sensitivity.py",
        "--cells",
        ANALYSIS / "nominal_reliability_cells.csv",
        "--absolute-summary",
        ANALYSIS / "nominal_reliability_summary.json",
        "--output-dir",
        ANALYSIS,
        "--qc-output",
        QC / "relative_loss_qc.json",
    )
    run(
        SCRIPTS / "analyze_loss_policy_ordering.py",
        "--pairs",
        ANALYSIS / "nominal_reliability_pairwise.csv",
        "--annotated-output",
        ANALYSIS / "nominal_reliability_loss_policy_pairwise.csv",
        "--summary-output",
        ANALYSIS / "nominal_reliability_loss_policy_summary.json",
        "--qc-output",
        QC / "loss_policy_qc.json",
    )
    run(
        SCRIPTS / "make_table1.py",
        "--input",
        ROOT / "data/reference_panel/primary_contrasts.csv",
        "--per-seed-root",
        ROOT / "data/reference_panel/per_seed",
        "--output",
        OUTPUTS / "Table_I.tex",
    )
    run(
        SCRIPTS / "make_fig2.py",
        "--nominal-reliability",
        ANALYSIS / "nominal_reliability_cells.csv",
        "--output",
        OUTPUTS / "Fig2_sensitivity_audit.pdf",
    )
    run(
        SCRIPTS / "verify_reported_numbers.py",
        "--output",
        OUTPUTS / "audit_report.json",
    )
    run(
        SCRIPTS / "make_sha256_manifest.py",
        "--root",
        OUTPUTS,
        "--output",
        "MANIFEST_SHA256.txt",
    )

    report = json.loads((OUTPUTS / "audit_report.json").read_text(encoding="utf-8"))
    status = report.get("status")
    print(f"Reproduction status: {status}")
    if args.check and status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
