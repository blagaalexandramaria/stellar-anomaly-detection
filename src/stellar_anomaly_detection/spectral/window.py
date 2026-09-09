"""Deterministic spectral-window and sampling diagnostics.

The formulas and defaults follow the frozen spectral-window definition.
Persistence fingerprints and internal workflow metadata are intentionally not
part of this public scientific callable surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class SamplingDiagnosticConfiguration:
    """Configuration for deterministic cadence and spectral-window diagnostics."""
    gap_policy_id: str = "median_cadence_multiplier_v1"
    gap_multiplier: float = 5.0
    spectral_window_formula_id: str = "normalized_timestamp_phasor_power_v1"
    chunk_size: int = 512


@dataclass(frozen=True)
class SpectralWindowResult:
    """Immutable spectral-window array and its sampling diagnostics."""
    spectral_window_power: NDArray[np.float64]
    cadence: dict[str, Any]
    gap: dict[str, Any]
    coverage: dict[str, Any]
    principal_peak: dict[str, Any]
    sidelobe: dict[str, Any]
    energy: dict[str, Any]
    symmetry: dict[str, Any]

    def __post_init__(self) -> None:
        value = np.asarray(self.spectral_window_power, dtype=np.float64).copy()
        value.setflags(write=False)
        object.__setattr__(self, "spectral_window_power", value)

    def to_dict(self) -> dict[str, Any]:
        """Return the frozen scientific fields in JSON-safe form."""
        return {
            "spectral_window_power": self.spectral_window_power.tolist(),
            "cadence": self.cadence,
            "gap": self.gap,
            "coverage": self.coverage,
            "principal_peak": self.principal_peak,
            "sidelobe": self.sidelobe,
            "energy": self.energy,
            "symmetry": self.symmetry,
        }


def _time(time: ArrayLike) -> NDArray[np.float64]:
    value = np.asarray(time, dtype=np.float64)
    if value.ndim != 1 or value.size < 3 or not np.all(np.isfinite(value)):
        raise ValueError("invalid_time_array")
    if np.any(np.diff(value) <= 0):
        raise ValueError("invalid_time_array")
    return value


def _frequency(frequency: ArrayLike) -> NDArray[np.float64]:
    value = np.asarray(frequency, dtype=np.float64)
    if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError("invalid_frequency_array")
    if np.any(value <= 0) or np.any(np.diff(value) <= 0):
        raise ValueError("invalid_frequency_array")
    return value


def compute_cadence_diagnostics(time: ArrayLike) -> dict[str, Any]:
    """Summarize cadence and baseline for strictly increasing times in days."""
    t = _time(time)
    delta = np.diff(t)
    mean = float(np.mean(delta))
    std = float(np.std(delta, ddof=0))
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("invalid_cadence")
    return {
        "baseline": float(t[-1] - t[0]),
        "baseline_unit": "day",
        "cadence_count": int(delta.size),
        "minimum_cadence": float(np.min(delta)),
        "maximum_cadence": float(np.max(delta)),
        "mean_cadence": mean,
        "median_cadence": float(np.median(delta)),
        "cadence_standard_deviation": std,
        "cadence_coefficient_of_variation": std / mean,
        "nominal_cadence": float(np.median(delta)),
        "standard_deviation_ddof": 0,
    }


def compute_gap_diagnostics(
    time: ArrayLike, cadence: dict[str, Any], gap_multiplier: float = 5.0
) -> dict[str, Any]:
    """Measure gaps exceeding a fixed multiple of the median cadence."""
    delta = np.diff(_time(time))
    nominal = cadence["nominal_cadence"]
    threshold = float(gap_multiplier * nominal)
    gaps = delta[delta > threshold]
    count = int(gaps.size)
    excess = float(np.sum(gaps - nominal)) if count else 0.0
    fraction = count / delta.size
    excess_fraction = excess / cadence["baseline"]
    if not 0 <= fraction <= 1 or not 0 <= excess_fraction <= 1:
        raise ValueError("invalid_gap_fraction")
    return {
        "gap_policy_id": "median_cadence_multiplier_v1",
        "gap_multiplier": float(gap_multiplier),
        "gap_threshold": threshold,
        "gap_condition": "delta_t_strictly_greater_than_threshold",
        "gap_count": count,
        "gap_interval_fraction": float(fraction),
        "largest_gap": float(np.max(gaps)) if count else None,
        "mean_gap": float(np.mean(gaps)) if count else None,
        "median_gap": float(np.median(gaps)) if count else None,
        "total_gap_duration": float(np.sum(gaps)) if count else 0.0,
        "total_gap_excess_duration": excess,
        "gap_excess_fraction": float(excess_fraction),
    }


def compute_sampling_coverage_diagnostics(
    cadence: dict[str, Any], gap: dict[str, Any]
) -> dict[str, Any]:
    """Derive effective coverage and regularity from cadence and gap summaries."""
    baseline = cadence["baseline"]
    observed = baseline - gap["total_gap_excess_duration"]
    duty = observed / baseline
    ratio = cadence["cadence_count"] * cadence["nominal_cadence"] / baseline
    regularity = 1 / (1 + cadence["cadence_coefficient_of_variation"])
    if not 0 <= duty <= 1:
        raise ValueError("invalid_duty_cycle")
    if not 0 < regularity <= 1:
        raise ValueError("invalid_sampling_regularity")
    return {
        "effective_observed_duration": float(observed),
        "effective_duty_cycle": float(duty),
        "nominal_sampling_coverage_ratio": float(ratio),
        "sampling_coverage_score": float(min(1, max(0, ratio))),
        "sampling_regularity_index": float(regularity),
    }


def compute_spectral_window_power(
    time: ArrayLike,
    frequency: ArrayLike,
    *,
    chunk_size: int = 512,
    negative: bool = False,
) -> NDArray[np.float64]:
    """Compute normalized timestamp-phasor power on an increasing frequency grid.

    Frequencies are processed in chunks without changing their order. Setting
    ``negative`` is used only to verify the expected frequency symmetry.
    """
    t, f = _time(time), _frequency(frequency)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    tau = t - t[0]
    output = np.empty(f.size, dtype=np.float64)
    sign = -1.0 if negative else 1.0
    for start in range(0, f.size, chunk_size):
        stop = min(start + chunk_size, f.size)
        phase = 2 * np.pi * sign * tau[:, None] * f[None, start:stop]
        cosine = np.cos(phase).sum(axis=0)
        sine = np.sin(phase).sum(axis=0)
        output[start:stop] = (cosine * cosine + sine * sine) / (t.size * t.size)
    if not np.all(np.isfinite(output)):
        raise ValueError("nonfinite_spectral_window")
    if np.any(output < 0):
        raise ValueError("negative_spectral_window_power")
    return output


def compute_window_principal_peak(frequency: ArrayLike, window: ArrayLike) -> dict[str, Any]:
    """Return the strongest window peak, resolving ties by first occurrence."""
    f = _frequency(frequency)
    w = np.asarray(window, dtype=np.float64)
    if w.shape != f.shape or not np.all(np.isfinite(w)):
        raise ValueError("invalid_spectral_window")
    index = int(np.argmax(w))
    power, freq = float(w[index]), float(f[index])
    return {
        "principal_window_index": index,
        "principal_window_frequency": freq,
        "principal_window_power": power,
        "principal_window_period": 1 / freq,
        "principal_window_tie_count": int(np.count_nonzero(w == power)),
        "tie_policy": "numpy_argmax_first_occurrence",
    }


def _local_maxima(window: NDArray[np.float64]) -> NDArray[np.int64]:
    return np.flatnonzero((window[1:-1] > window[:-2]) & (window[1:-1] >= window[2:])) + 1


def compute_window_sidelobe_diagnostic(
    frequency: ArrayLike,
    window: ArrayLike,
    principal: dict[str, Any],
    baseline: float,
) -> dict[str, Any]:
    """Return the strongest local maximum outside one baseline resolution bin."""
    f = _frequency(frequency)
    w = np.asarray(window, dtype=np.float64)
    local = _local_maxima(w)
    resolution = 1 / baseline
    principal_frequency = principal["principal_window_frequency"]
    candidates = local[
        (f[local] < principal_frequency - resolution)
        | (f[local] > principal_frequency + resolution)
    ]
    if candidates.size:
        powers = w[candidates]
        index = int(candidates[int(np.argmax(powers))])
        frequency_value, power = float(f[index]), float(w[index])
        ratio = power / principal["principal_window_power"]
    else:
        index = frequency_value = power = ratio = None
    return {
        "window_local_maximum_count": int(local.size),
        "frequency_resolution": float(resolution),
        "principal_exclusion_interval": [
            float(principal_frequency - resolution),
            float(principal_frequency + resolution),
        ],
        "sidelobe_index": index,
        "maximum_sidelobe_frequency": frequency_value,
        "maximum_sidelobe_power": power,
        "maximum_sidelobe_ratio": ratio,
        "tie_policy": "numpy_argmax_first_occurrence",
    }


def compute_window_energy_diagnostic(frequency: ArrayLike, window: ArrayLike) -> dict[str, Any]:
    """Integrate window power on the supplied positive-frequency grid."""
    f, w = _frequency(frequency), np.asarray(window, dtype=np.float64)
    energy = float(np.trapezoid(w, f))
    if not np.isfinite(energy) or energy < 0:
        raise ValueError("window_energy_failure")
    return {
        "integration_method_id": "positive_grid_trapezoidal_window_energy_v1",
        "window_energy": energy,
        "window_mean_power": float(np.mean(w)),
        "window_median_power": float(np.median(w)),
        "window_standard_deviation": float(np.std(w, ddof=0)),
        "window_maximum_power": float(np.max(w)),
        "window_minimum_power": float(np.min(w)),
    }


def compute_window_symmetry_diagnostic(
    time: ArrayLike,
    frequency: ArrayLike,
    positive: ArrayLike,
    chunk_size: int = 512,
) -> dict[str, Any]:
    """Quantify numerical symmetry between positive and negative frequencies."""
    positive_array = np.asarray(positive, dtype=np.float64)
    negative = compute_spectral_window_power(
        time, frequency, chunk_size=chunk_size, negative=True
    )
    error = np.abs(positive_array - negative)
    return {
        "window_symmetry_max_absolute_error": float(np.max(error)),
        "window_symmetry_mean_absolute_error": float(np.mean(error)),
        "window_symmetry_rms_error": float(np.sqrt(np.mean(error * error))),
        "negative_frequency_array_persisted": False,
    }


def analyze_spectral_window(
    time: ArrayLike,
    frequency: ArrayLike,
    configuration: SamplingDiagnosticConfiguration | None = None,
) -> SpectralWindowResult:
    """Calculate the frozen public spectral-window diagnostics in memory."""
    configuration = configuration or SamplingDiagnosticConfiguration()
    cadence = compute_cadence_diagnostics(time)
    gap = compute_gap_diagnostics(time, cadence, configuration.gap_multiplier)
    coverage = compute_sampling_coverage_diagnostics(cadence, gap)
    window = compute_spectral_window_power(time, frequency, chunk_size=configuration.chunk_size)
    principal = compute_window_principal_peak(frequency, window)
    sidelobe = compute_window_sidelobe_diagnostic(
        frequency, window, principal, cadence["baseline"]
    )
    energy = compute_window_energy_diagnostic(frequency, window)
    symmetry = compute_window_symmetry_diagnostic(
        time, frequency, window, configuration.chunk_size
    )
    return SpectralWindowResult(
        window,
        cadence,
        gap,
        coverage,
        principal,
        sidelobe,
        energy,
        symmetry,
    )
