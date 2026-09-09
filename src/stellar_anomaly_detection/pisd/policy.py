"""Frozen PISD harmonic and stability constants.

Values are migrated directly from the final execution policy used by the
scientific authority. They are not user-tuned publication defaults.
"""

from __future__ import annotations

import hashlib

HARMONIC_EPSILON = 1e-12
HARMONIC_GRID_TOLERANCE_FACTOR = 1.0
HARMONIC_TOLERANCE_FACTOR = 1.0
MAX_FUNDAMENTAL_CANDIDATES = 1
MAX_HARMONIC_ORDER = 10

DOMINANT_PEAK_RAYLEIGH_FACTOR = 1.0
STABILITY_EPSILON = 1e-12
STABILITY_FALLBACK_NOISE_FRACTION = 0.001
STABILITY_NEGATIVE_POWER_TOLERANCE = 1e-12
STABILITY_NOISE_SCALE_FACTOR = 1.0
STABILITY_N_PERTURBATIONS = 100
STABILITY_RANDOM_STATE = 42
POLICY_VERSION = "1.0.0"


def derive_observation_stability_seed(
    canonical_object_id: str,
    observation_id: str,
    policy_version: str = POLICY_VERSION,
) -> int:
    """Derive the frozen per-observation SHA-256 child seed."""
    if not canonical_object_id or not observation_id or not policy_version:
        raise ValueError("seed identity fields must be non-empty")
    material = f"42\n{canonical_object_id}\n{observation_id}\n{policy_version}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
