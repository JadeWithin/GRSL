#!/usr/bin/env python3
"""Validate the two-stage SRF forward model before classifier experiments.

The reference path convolves a high-resolution spectrum directly with the
effective target response.  The proxy path first observes the spectrum with a
6 nm source sensor on the real WHU-Hi wavelength grid and then applies the
additional response kernel needed to reach the target FWHM.

This script deliberately has no model-training dependency and requires only
NumPy from outside the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
ENVI_LIST_RE = re.compile(r"(?P<key>[A-Za-z_ ]+)\s*=\s*\{(?P<value>.*?)\}", re.DOTALL)


@dataclass(frozen=True)
class SpectrumRecord:
    sample_id: str
    class_name: str
    spectrum_path: Path
    wavelength_path: Path
    wavelength_unit: str


def trapezoid_widths(grid: np.ndarray) -> np.ndarray:
    """Return integration widths equivalent to np.trapezoid on a 1-D grid."""
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("grid must be one-dimensional with at least two samples")
    delta = np.diff(grid)
    if np.any(delta <= 0):
        raise ValueError("grid must be strictly increasing")
    widths = np.empty_like(grid)
    widths[0] = delta[0] / 2.0
    widths[-1] = delta[-1] / 2.0
    widths[1:-1] = (delta[:-1] + delta[1:]) / 2.0
    return widths


def gaussian_response_matrix(
    grid_nm: np.ndarray,
    centers_nm: np.ndarray,
    fwhm_nm: float,
) -> np.ndarray:
    """Create area-normalized Gaussian response rows on an arbitrary grid."""
    if fwhm_nm <= 0:
        raise ValueError("FWHM must be positive")
    grid_nm = np.asarray(grid_nm, dtype=np.float64)
    centers_nm = np.asarray(centers_nm, dtype=np.float64)
    sigma = fwhm_nm * FWHM_TO_SIGMA
    z = (grid_nm[None, :] - centers_nm[:, None]) / sigma
    response = np.exp(-0.5 * z * z)
    response *= trapezoid_widths(grid_nm)[None, :]
    denominator = response.sum(axis=1, keepdims=True)
    if np.any(denominator <= np.finfo(np.float64).tiny):
        raise ValueError("a response has no support on the supplied grid")
    return response / denominator


def apply_response(
    spectrum: np.ndarray,
    grid_nm: np.ndarray,
    centers_nm: np.ndarray,
    fwhm_nm: float,
) -> np.ndarray:
    spectrum = np.asarray(spectrum, dtype=np.float64)
    if spectrum.shape != np.asarray(grid_nm).shape:
        raise ValueError("spectrum and wavelength grid must have identical shapes")
    return gaussian_response_matrix(grid_nm, centers_nm, fwhm_nm) @ spectrum


def effective_to_additional_fwhm(effective_fwhm_nm: float, source_fwhm_nm: float) -> float:
    variance = effective_fwhm_nm**2 - source_fwhm_nm**2
    if variance <= 0:
        raise ValueError(
            f"effective FWHM {effective_fwhm_nm} must exceed source FWHM {source_fwhm_nm}"
        )
    return math.sqrt(variance)


def direct_path(
    highres_spectrum: np.ndarray,
    highres_grid_nm: np.ndarray,
    target_centers_nm: np.ndarray,
    effective_fwhm_nm: float,
) -> np.ndarray:
    return apply_response(
        highres_spectrum,
        highres_grid_nm,
        target_centers_nm,
        effective_fwhm_nm,
    )


def two_stage_path(
    highres_spectrum: np.ndarray,
    highres_grid_nm: np.ndarray,
    source_centers_nm: np.ndarray,
    source_fwhm_nm: float,
    target_centers_nm: np.ndarray,
    effective_fwhm_nm: float,
) -> np.ndarray:
    source_observation = apply_response(
        highres_spectrum,
        highres_grid_nm,
        source_centers_nm,
        source_fwhm_nm,
    )
    additional_fwhm = effective_to_additional_fwhm(effective_fwhm_nm, source_fwhm_nm)
    return apply_response(
        source_observation,
        source_centers_nm,
        target_centers_nm,
        additional_fwhm,
    )


def relative_nrmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    rms_error = float(np.sqrt(np.mean((estimate - reference) ** 2)))
    rms_reference = float(np.sqrt(np.mean(reference**2)))
    return rms_error / max(rms_reference, np.finfo(np.float64).eps)


def spectral_angle_deg(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    denominator = float(np.linalg.norm(estimate) * np.linalg.norm(reference))
    if denominator <= np.finfo(np.float64).eps:
        return float("nan")
    cosine = float(np.dot(estimate, reference) / denominator)
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def perturbation_error_ratio(
    proxy_perturbed: np.ndarray,
    proxy_nominal: np.ndarray,
    direct_perturbed: np.ndarray,
    direct_nominal: np.ndarray,
) -> float:
    proxy_delta = np.asarray(proxy_perturbed) - np.asarray(proxy_nominal)
    direct_delta = np.asarray(direct_perturbed) - np.asarray(direct_nominal)
    numerator = float(np.linalg.norm(proxy_delta - direct_delta))
    denominator = float(np.linalg.norm(direct_delta))
    return numerator / max(denominator, np.finfo(np.float64).eps)


def _numeric_vector(path: Path) -> np.ndarray:
    """Read one-column USGS-style ASCII data, with a conservative fallback."""
    try:
        values = np.loadtxt(path, comments="#", dtype=np.float64)
        if values.ndim == 2:
            values = values[:, -1]
        return np.asarray(values, dtype=np.float64).reshape(-1)
    except ValueError:
        parsed: list[float] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            tokens = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", line)
            if tokens:
                parsed.append(float(tokens[-1]))
        if not parsed:
            raise ValueError(f"no numeric values found in {path}")
        return np.asarray(parsed, dtype=np.float64)


def load_spectrum(record: SpectrumRecord) -> tuple[np.ndarray, np.ndarray]:
    wavelength = _numeric_vector(record.wavelength_path)
    spectrum = _numeric_vector(record.spectrum_path)
    if wavelength.size != spectrum.size:
        raise ValueError(
            f"{record.sample_id}: wavelength count {wavelength.size} != spectrum count {spectrum.size}"
        )
    unit = record.wavelength_unit.strip().lower()
    if unit in {"um", "micron", "microns", "micrometer", "micrometers"}:
        wavelength = wavelength * 1000.0
    elif unit != "nm":
        raise ValueError(f"unsupported wavelength unit: {record.wavelength_unit}")
    valid = (
        np.isfinite(wavelength)
        & np.isfinite(spectrum)
        & (spectrum > -1.0e30)
        & (wavelength >= 390.0)
        & (wavelength <= 1010.0)
    )
    wavelength = wavelength[valid]
    spectrum = spectrum[valid]
    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    spectrum = spectrum[order]
    unique = np.concatenate(([True], np.diff(wavelength) > 0))
    wavelength = wavelength[unique]
    spectrum = spectrum[unique]
    if wavelength.size < 100:
        raise ValueError(f"{record.sample_id}: only {wavelength.size} usable VNIR samples")
    return wavelength, spectrum


def parse_envi_wavelengths(header_path: Path) -> np.ndarray:
    text = header_path.read_text(encoding="utf-8", errors="replace")
    matches = {m.group("key").strip().lower(): m.group("value") for m in ENVI_LIST_RE.finditer(text)}
    if "wavelength" not in matches:
        raise ValueError(f"no wavelength list in ENVI header {header_path}")
    values = [float(token) for token in re.split(r"[\s,]+", matches["wavelength"].strip()) if token]
    wavelength = np.asarray(values, dtype=np.float64)
    if np.nanmedian(wavelength) < 10.0:
        wavelength *= 1000.0
    if wavelength.size < 2 or np.any(np.diff(wavelength) <= 0):
        raise ValueError("ENVI wavelength list is not strictly increasing")
    return wavelength


def load_manifest(path: Path) -> list[SpectrumRecord]:
    records: list[SpectrumRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_id",
            "class_name",
            "spectrum_path",
            "wavelength_path",
            "wavelength_unit",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"manifest must contain columns: {sorted(required)}")
        for row in reader:
            records.append(
                SpectrumRecord(
                    sample_id=row["sample_id"],
                    class_name=row["class_name"],
                    spectrum_path=(path.parent / row["spectrum_path"]).resolve(),
                    wavelength_path=(path.parent / row["wavelength_path"]).resolve(),
                    wavelength_unit=row["wavelength_unit"],
                )
            )
    if not records:
        raise ValueError("manifest is empty")
    return records


def summarize(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    conditions = sorted({str(row["condition"]) for row in rows})
    summary: dict[str, object] = {"sample_count": len({str(row["sample_id"]) for row in rows})}
    condition_summary: dict[str, object] = {}
    for condition in conditions:
        subset = [row for row in rows if row["condition"] == condition]
        q = np.asarray([float(row["q"]) for row in subset], dtype=np.float64)
        nrmse = np.asarray([float(row["nrmse"]) for row in subset], dtype=np.float64)
        sam = np.asarray([float(row["sam_deg"]) for row in subset], dtype=np.float64)
        condition_summary[condition] = {
            "n": len(subset),
            "q_median": float(np.nanmedian(q)),
            "q_p95": float(np.nanpercentile(q, 95)),
            "nrmse_median": float(np.nanmedian(nrmse)),
            "sam_deg_median": float(np.nanmedian(sam)),
            "q_gate_pass": bool(np.nanmedian(q) <= 0.25 and np.nanpercentile(q, 95) <= 0.50),
        }
    summary["conditions"] = condition_summary
    return summary


def evaluate(
    records: Iterable[SpectrumRecord],
    source_centers_nm: np.ndarray,
    source_fwhm_nm: float,
    target_centers_nm: np.ndarray,
    nominal_fwhm_nm: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    perturbations = [
        ("center_minus_2p2_nm", -2.2, nominal_fwhm_nm),
        ("center_plus_2p2_nm", 2.2, nominal_fwhm_nm),
        ("fwhm_12_nm", 0.0, 12.0),
        ("fwhm_18_nm", 0.0, 18.0),
    ]
    for record in records:
        grid_nm, spectrum = load_spectrum(record)
        if grid_nm[0] > source_centers_nm[0] or grid_nm[-1] < source_centers_nm[-1]:
            raise ValueError(
                f"{record.sample_id}: coverage {grid_nm[0]:.2f}-{grid_nm[-1]:.2f} nm "
                f"does not span source centers {source_centers_nm[0]:.2f}-{source_centers_nm[-1]:.2f} nm"
            )
        direct_nominal = direct_path(spectrum, grid_nm, target_centers_nm, nominal_fwhm_nm)
        proxy_nominal = two_stage_path(
            spectrum,
            grid_nm,
            source_centers_nm,
            source_fwhm_nm,
            target_centers_nm,
            nominal_fwhm_nm,
        )
        for condition, center_shift_nm, effective_fwhm_nm in perturbations:
            perturbed_centers = target_centers_nm + center_shift_nm
            direct_perturbed = direct_path(
                spectrum,
                grid_nm,
                perturbed_centers,
                effective_fwhm_nm,
            )
            proxy_perturbed = two_stage_path(
                spectrum,
                grid_nm,
                source_centers_nm,
                source_fwhm_nm,
                perturbed_centers,
                effective_fwhm_nm,
            )
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "class_name": record.class_name,
                    "condition": condition,
                    "q": perturbation_error_ratio(
                        proxy_perturbed,
                        proxy_nominal,
                        direct_perturbed,
                        direct_nominal,
                    ),
                    "nrmse": relative_nrmse(proxy_perturbed, direct_perturbed),
                    "sam_deg": spectral_angle_deg(proxy_perturbed, direct_perturbed),
                    "nominal_nrmse": relative_nrmse(proxy_nominal, direct_nominal),
                    "nominal_sam_deg": spectral_angle_deg(proxy_nominal, direct_nominal),
                }
            )
    return rows


def write_outputs(rows: Sequence[dict[str, object]], output_dir: Path, metadata: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with (output_dir / "per_spectrum_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    result = {"metadata": metadata, "summary": summarize(rows)}
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-envi-header", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-fwhm-nm", type=float, default=6.0)
    parser.add_argument("--nominal-fwhm-nm", type=float, default=15.0)
    parser.add_argument("--target-start-nm", type=float, default=430.0)
    parser.add_argument("--target-stop-nm", type=float, default=966.0)
    parser.add_argument("--target-step-nm", type=float, default=8.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_centers = parse_envi_wavelengths(args.source_envi_header)
    target_centers = np.arange(
        args.target_start_nm,
        args.target_stop_nm + args.target_step_nm / 2.0,
        args.target_step_nm,
        dtype=np.float64,
    )
    records = load_manifest(args.manifest)
    rows = evaluate(
        records,
        source_centers_nm=source_centers,
        source_fwhm_nm=args.source_fwhm_nm,
        target_centers_nm=target_centers,
        nominal_fwhm_nm=args.nominal_fwhm_nm,
    )
    metadata = {
        "manifest": str(args.manifest.resolve()),
        "source_envi_header": str(args.source_envi_header.resolve()),
        "source_center_count": int(source_centers.size),
        "source_center_min_nm": float(source_centers.min()),
        "source_center_max_nm": float(source_centers.max()),
        "source_fwhm_nm": args.source_fwhm_nm,
        "target_center_count": int(target_centers.size),
        "target_center_min_nm": float(target_centers.min()),
        "target_center_max_nm": float(target_centers.max()),
        "target_step_nm": args.target_step_nm,
        "nominal_effective_fwhm_nm": args.nominal_fwhm_nm,
    }
    write_outputs(rows, args.output_dir, metadata)
    print(json.dumps({"output_dir": str(args.output_dir), "summary": summarize(rows)}, indent=2))


if __name__ == "__main__":
    main()
