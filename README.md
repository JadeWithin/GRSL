# Center-Wavelength Shift Sensitivity in Hyperspectral Classification

This repository accompanies the manuscript **"Center-Wavelength Shift Sensitivity in Hyperspectral Classification: A Paired Virtual-Response Audit."**

## Review-stage partial release

This public review-stage snapshot intentionally provides a partial, non-core code release for methodological inspection.

Currently available:

- operator-level spectral-response forward-model validation utilities;
- post hoc nominal-performance versus shift-loss ordering analysis;
- relative-loss and shared-loss-policy sensitivity analyses; and
- deterministic SHA-256 manifest generation.

The released analysis scripts expose their input schemas and command-line interfaces, but the complete derived tables are not included in the review-stage snapshot.

Not included at this stage:

- end-to-end model training and checkpoint selection;
- the complete shifted-inference and experiment-orchestration pipeline;
- full model and response-geometry configurations;
- fixed split files, complete derived tables, or trained weights.

The core training and shifted-inference implementation, fixed split hashes, derived analysis tables, and complete reproducibility instructions will be released after manuscript acceptance. Raw WHU-Hi imagery and trained model weights will not be redistributed; the dataset remains available from its official source.

## Current scripts

- `scripts/srf_forward_smoke.py`
- `scripts/analyze_nominal_reliability_discordance.py`
- `scripts/analyze_relative_loss_sensitivity.py`
- `scripts/analyze_loss_policy_ordering.py`
- `scripts/make_sha256_manifest.py`

The current snapshot requires Python 3.10+ and NumPy for the numerical analysis utilities.