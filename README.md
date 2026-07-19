# Center-Wavelength Shift Sensitivity in Hyperspectral Classification

This repository accompanies the manuscript **“Center-Wavelength Shift Sensitivity in Hyperspectral Classification: A Paired Virtual-Response Audit.”**

## What this capsule reproduces

This is a closed **analysis-level audit reproduction capsule**. From the released fixed derived outputs, one command:

- recomputes the 45 within-scene nominal-versus-reliability configuration comparisons;
- repeats the relative-loss and shared-loss-policy sensitivity checks;
- verifies the primary, dose, probabilistic, decision-transition, spatial-block, response-geometry, parameter-count, and direct-integration proxy summaries reported in the manuscript;
- recomputes all 36 primary percentile intervals from the released 2,000-replicate bootstrap arrays;
- regenerates Table I and Fig. 2; and
- writes a SHA-256 manifest for every generated output.

The capsule intentionally does **not** claim end-to-end retraining reproduction. Raw WHU-Hi imagery, pixel-level predictions, trained weights, and the training/shifted-inference orchestration are not redistributed. The released scope is sufficient to reproduce and check the manuscript’s audit-level numerical summaries and displayed evidence from fixed model outputs.

## Quick start

Requirements: Python 3.10 or later.

```bash
python -m pip install -r requirements.txt
python scripts/reproduce_audit.py --check
```

A successful run ends with:

```text
Reproduction status: PASS
```

Key generated files are:

- `outputs/audit_report.json`
- `outputs/Table_I.tex`
- `outputs/Fig2_sensitivity_audit.pdf`
- `outputs/Fig2_sensitivity_audit.png`
- `outputs/MANIFEST_SHA256.txt`

## Released evidence

- `data/reference_panel/`: original-response primary contrasts, dose summaries, all evaluated conditions, five-seed metrics, and the primary block-bootstrap arrays.
- `data/reference_secondary/`: probabilistic endpoints and block-size sensitivity.
- `data/geometry_panel/`: the 108 response-geometry contrasts.
- `data/reproducibility/`: five-seed variability and trainable-parameter counts by virtual response.
- `data/proxy_validation/`: per-spectrum direct-integration proxy checks for the three virtual responses.
- `scripts/`: ordering analyses, table/figure generation, numerical verification, response-operator smoke testing, and deterministic manifest generation.

## Interpretation boundary

The released responses are designed virtual operators, not measurements from three named instruments. All numerical conclusions are conditional on the audited WHU-Hi scenes, spatial split, trained configurations, seeds, and declared perturbation envelope. The profile is a conditional system diagnostic, not a universal acceptance rule.

WHU-Hi is available from its official source cited in the manuscript. Dataset licensing and redistribution terms remain with the dataset provider.
