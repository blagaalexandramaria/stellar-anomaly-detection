#!/usr/bin/env python3
"""Regenerate the deterministic software-demonstration light curve."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

SEED = 20260821
SAMPLES = 512
BASELINE = 30.0
FUNDAMENTAL_FREQUENCY = 1.25
FUNDAMENTAL_AMPLITUDE = 0.08
HARMONIC_AMPLITUDE = 0.025
NOISE_STANDARD_DEVIATION = 0.008
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "data"


def generate() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic time, normalized flux, and nominal uncertainty."""
    rng = np.random.default_rng(SEED)
    time = np.linspace(0.0, BASELINE, SAMPLES, dtype=np.float64)
    time[1:-1] += rng.uniform(-0.01, 0.01, SAMPLES - 2)
    flux = (
        1.0
        + FUNDAMENTAL_AMPLITUDE
        * np.sin(2.0 * np.pi * FUNDAMENTAL_FREQUENCY * time)
        + HARMONIC_AMPLITUDE
        * np.sin(4.0 * np.pi * FUNDAMENTAL_FREQUENCY * time + 0.35)
        + rng.normal(0.0, NOISE_STANDARD_DEVIATION, SAMPLES)
    )
    flux_error = np.full(SAMPLES, NOISE_STANDARD_DEVIATION, dtype=np.float64)
    return time, flux, flux_error


def main() -> int:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIRECTORY / "synthetic_demo_light_curve.csv"
    provenance_path = OUTPUT_DIRECTORY / "synthetic_demo_provenance.json"
    time, flux, flux_error = generate()

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("time", "flux", "flux_error"))
        writer.writerows(
            (
                f"{sample_time:.10f}",
                f"{sample_flux:.10f}",
                f"{sample_error:.10f}",
            )
            for sample_time, sample_flux, sample_error in zip(
                time, flux, flux_error, strict=True
            )
        )

    provenance = {
        "dataset_name": "synthetic_demo_light_curve",
        "dataset_type": "synthetic",
        "purpose": "software demonstration",
        "publication_corpus_member": False,
        "observational_data": False,
        "random_seed": SEED,
        "generation_description": (
            "Mildly irregular times over 30 arbitrary time units; one sinusoid, "
            "one weaker second harmonic, and deterministic Gaussian noise."
        ),
        "generation_parameters": {
            "samples": SAMPLES,
            "baseline": BASELINE,
            "fundamental_frequency": FUNDAMENTAL_FREQUENCY,
            "fundamental_amplitude": FUNDAMENTAL_AMPLITUDE,
            "second_harmonic_amplitude": HARMONIC_AMPLITUDE,
            "noise_standard_deviation": NOISE_STANDARD_DEVIATION,
        },
        "number_of_samples": SAMPLES,
        "columns": ["time", "flux", "flux_error"],
        "units": {
            "time": "arbitrary time unit",
            "flux": "normalized dimensionless flux",
            "flux_error": "normalized dimensionless flux",
        },
        "license_relationship": (
            "Project-created synthetic demonstration data distributed with "
            "this repository under the repository MIT License."
        ),
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {csv_path.relative_to(OUTPUT_DIRECTORY.parent)}")
    print(f"Wrote {provenance_path.relative_to(OUTPUT_DIRECTORY.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
