"""Validated Lomb--Scargle analysis for astronomical light curves."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
import json
import time as timer
from typing import Any, Literal

import astropy
import numpy as np
from astropy.timeseries import LombScargle
from numpy.typing import ArrayLike, NDArray

from stellar_anomaly_detection.constants import (
    FALSE_ALARM_PROBABILITY,
    MAX_FREQUENCY,
    MIN_FREQUENCY,
    SAMPLES_PER_PEAK,
)

Normalization = Literal["standard", "model", "log", "psd"]
ComputationMethod = Literal["auto", "slow", "cython", "chi2", "fast", "fastchi2", "scipy"]
FalseAlarmMethod = Literal["baluev", "davies", "naive", "bootstrap"]
FluxRepresentation = Literal["raw", "global_relative", "segment_relative", "unspecified"]

_USE_ANALYZER_DEFAULT = object()


@dataclass(frozen=True)
class LombScargleResult:
    """Immutable, JSON-serializable result of one Lomb--Scargle analysis."""

    frequency: NDArray[np.float64]
    power: NDArray[np.float64]
    dominant_frequency: float
    dominant_period: float
    dominant_power: float
    false_alarm_probability: float | None
    false_alarm_power_level: float | None
    false_alarm_probability_threshold: float | None
    is_significant: bool | None
    false_alarm_method: str | None
    minimum_frequency: float
    maximum_frequency: float
    frequency_step: float
    frequency_samples: int
    samples_per_peak: int
    normalization: str
    fit_mean: bool
    center_data: bool
    nterms: int
    computation_method_requested: str
    computation_method_used: str | None
    n_observations: int
    n_distinct_times: int
    baseline_days: float
    median_cadence_days: float | None
    flux_representation: str
    used_flux_error: bool
    runtime_seconds: float
    warnings: tuple[str, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        frequency = np.array(self.frequency, dtype=np.float64, copy=True)
        power = np.array(self.power, dtype=np.float64, copy=True)
        if frequency.ndim != 1 or power.ndim != 1:
            raise ValueError("frequency and power must be 1D arrays.")
        if frequency.size < 2 or power.size != frequency.size:
            raise ValueError("frequency and power must have equal length of at least 2.")
        if not np.all(np.isfinite(frequency)) or not np.all(np.isfinite(power)):
            raise ValueError("frequency and power must contain only finite values.")
        if np.any(frequency <= 0) or np.any(np.diff(frequency) <= 0):
            raise ValueError("frequency must be strictly positive and increasing.")

        required_positive = {
            "dominant_frequency": self.dominant_frequency,
            "dominant_period": self.dominant_period,
            "frequency_step": self.frequency_step,
            "baseline_days": self.baseline_days,
        }
        for name, value in required_positive.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in (
            "dominant_power",
            "minimum_frequency",
            "maximum_frequency",
            "runtime_seconds",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or (name != "dominant_power" and value < 0):
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.maximum_frequency <= self.minimum_frequency:
            raise ValueError("maximum_frequency must exceed minimum_frequency.")
        if self.frequency_samples != frequency.size:
            raise ValueError("frequency_samples must match frequency length.")
        if self.n_observations < 3 or self.n_distinct_times != self.n_observations:
            raise ValueError("observation counts are inconsistent.")
        if self.median_cadence_days is not None and (
            not np.isfinite(self.median_cadence_days) or self.median_cadence_days <= 0
        ):
            raise ValueError("median_cadence_days must be finite and positive.")
        for name in ("false_alarm_probability", "false_alarm_probability_threshold"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be finite and in [0, 1].")
        if self.false_alarm_power_level is not None and not np.isfinite(
            self.false_alarm_power_level
        ):
            raise ValueError("false_alarm_power_level must be finite.")
        if not np.isfinite(self.dominant_power):
            raise ValueError("dominant_power must be finite.")
        if not isinstance(self.warnings, tuple) or not all(
            isinstance(item, str) for item in self.warnings
        ):
            raise TypeError("warnings must be a tuple of strings.")

        metadata = deepcopy(self.metadata)
        try:
            json.dumps(metadata, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON-safe and finite.") from exc
        frequency.setflags(write=False)
        power.setflags(write=False)
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "power", power)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self, *, include_arrays: bool = True) -> dict[str, Any]:
        """Return JSON-safe data; omit frequency and power when requested."""
        result: dict[str, Any] = {}
        for field in fields(self):
            if field.name in {"frequency", "power"}:
                if include_arrays:
                    result[field.name] = getattr(self, field.name).tolist()
                continue
            value = getattr(self, field.name)
            result[field.name] = list(value) if field.name == "warnings" else deepcopy(value)
        json.dumps(result, allow_nan=False)
        return result


@dataclass(frozen=True)
class _ValidatedInput:
    time: NDArray[np.float64]
    flux: NDArray[np.float64]
    flux_error: NDArray[np.float64] | None
    input_was_sorted: bool
    n_observations: int
    n_distinct_times: int
    baseline_days: float
    median_cadence_days: float


class LombScargleAnalyzer:
    """Compute validated single-term or multi-term Lomb--Scargle spectra."""

    _VALID_NORMALIZATIONS = frozenset({"standard", "model", "log", "psd"})
    _VALID_COMPUTATION_METHODS = frozenset(
        {"auto", "slow", "cython", "chi2", "fast", "fastchi2", "scipy"}
    )
    _VALID_FALSE_ALARM_METHODS = frozenset({"baluev", "davies", "naive", "bootstrap"})
    _VALID_FLUX_REPRESENTATIONS = frozenset(
        {"raw", "global_relative", "segment_relative", "unspecified"}
    )
    _LARGE_GRID_WARNING_THRESHOLD = 1_000_000
    _GRID_UNIFORMITY_RTOL = 1e-10
    _GRID_UNIFORMITY_ATOL = 1e-14

    def __init__(
        self,
        minimum_frequency: float = MIN_FREQUENCY,
        maximum_frequency: float = MAX_FREQUENCY,
        samples_per_peak: int = SAMPLES_PER_PEAK,
        false_alarm_probability: float | None = FALSE_ALARM_PROBABILITY,
        normalization: Normalization = "standard",
        center_data: bool = True,
        fit_mean: bool = True,
        nterms: int = 1,
        minimum_observations: int = 3,
        computation_method: ComputationMethod = "auto",
    ) -> None:
        self.minimum_frequency = self._validate_frequency_bound(
            minimum_frequency, "minimum_frequency"
        )
        self.maximum_frequency = self._validate_frequency_bound(
            maximum_frequency, "maximum_frequency"
        )
        if self.maximum_frequency <= self.minimum_frequency:
            raise ValueError("maximum_frequency must be greater than minimum_frequency.")
        self.samples_per_peak = self._validate_positive_integer(
            samples_per_peak, "samples_per_peak"
        )
        self.false_alarm_probability = self._validate_probability(false_alarm_probability)
        self.normalization = self._validate_normalization(normalization)
        self.center_data = bool(center_data)
        self.fit_mean = bool(fit_mean)
        self.nterms = self._validate_positive_integer(nterms, "nterms")
        self.minimum_observations = self._validate_positive_integer(
            minimum_observations, "minimum_observations"
        )
        if self.minimum_observations < 3:
            raise ValueError("minimum_observations must be at least 3.")
        self.computation_method = self._validate_computation_method(computation_method)

    def validate_input(
        self,
        time: ArrayLike,
        flux: ArrayLike,
        flux_error: ArrayLike | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
        """Return owned float64 copies sorted stably by strictly unique time."""
        validated = self._validate_input(time, flux, flux_error)
        return validated.time, validated.flux, validated.flux_error

    def analyze(
        self,
        time: ArrayLike,
        flux: ArrayLike,
        flux_error: ArrayLike | None = None,
        *,
        flux_representation: FluxRepresentation = "unspecified",
        minimum_frequency: float | None = None,
        maximum_frequency: float | None = None,
        samples_per_peak: int | None = None,
        normalization: Normalization | None = None,
        computation_method: ComputationMethod | None = None,
        false_alarm_probability: float | None | object = _USE_ANALYZER_DEFAULT,
        false_alarm_method: FalseAlarmMethod = "baluev",
    ) -> LombScargleResult:
        """Run one complete Lomb--Scargle analysis.

        Tied maximum powers are resolved by ``numpy.argmax``, which selects the
        first occurrence. The median-cadence Nyquist value is diagnostic only:
        irregular sampling has no single strict Nyquist limit.
        """
        start = timer.perf_counter()
        fmin, fmax, spp, norm, method, fap_threshold = self._resolve_options(
            minimum_frequency,
            maximum_frequency,
            samples_per_peak,
            normalization,
            computation_method,
            false_alarm_probability,
        )
        representation = self._validate_flux_representation(flux_representation)
        fap_method = self._validate_false_alarm_method(false_alarm_method)
        validated = self._validate_input(time, flux, flux_error)
        model = self._build_model(validated, norm)

        try:
            frequency, power = model.autopower(
                method=method,
                minimum_frequency=fmin,
                maximum_frequency=fmax,
                samples_per_peak=spp,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Astropy could not compute the periodogram with method {method!r}."
            ) from exc
        frequency_array, power_array = self._validate_periodogram(frequency, power)
        frequency_step, grid_uniform = self._grid_diagnostics(frequency_array)

        messages: list[str] = []
        if not grid_uniform:
            messages.append("The frequency grid is not uniform within numerical tolerance.")
        self._append_large_grid_warning(messages, frequency_array.size)

        dominant_index = int(np.argmax(power_array))
        at_boundary = dominant_index in {0, frequency_array.size - 1}
        if at_boundary:
            messages.append(
                "The dominant maximum is at a frequency-grid boundary; "
                "the configured frequency interval may be insufficient."
            )
        dominant_frequency = float(frequency_array[dominant_index])
        dominant_power = float(power_array[dominant_index])

        median_nyquist = 0.5 / validated.median_cadence_days
        if fmax > median_nyquist:
            messages.append(
                "maximum_frequency exceeds the median-cadence Nyquist diagnostic; "
                "this is not a strict Nyquist limit for irregular sampling."
            )

        fap_value: float | None = None
        fap_power_level: float | None = None
        significant: bool | None = None
        result_fap_method: str | None = None
        if fap_threshold is not None:
            result_fap_method = fap_method
            if self.nterms != 1:
                messages.append(
                    "False-alarm probability and power level were not computed: "
                    "Astropy false-alarm approximations require nterms=1."
                )
            else:
                fap_value, fap_power_level = self._compute_false_alarm(
                    model, dominant_power, fap_threshold, fap_method, fmin, fmax, spp
                )
                significant = bool(fap_value <= fap_threshold)

        runtime_seconds = float(timer.perf_counter() - start)
        method_used = None if method == "auto" else method
        metadata: dict[str, Any] = {
            "astropy_version": str(astropy.__version__),
            "frequency_grid_source": "astropy.autopower",
            "median_nyquist_frequency": float(median_nyquist),
            "dominant_index": dominant_index,
            "maximum_at_frequency_boundary": at_boundary,
            "grid_uniform": grid_uniform,
            "grid_uniformity_tolerance": {
                "relative": self._GRID_UNIFORMITY_RTOL,
                "absolute": self._GRID_UNIFORMITY_ATOL,
            },
            "input_was_sorted": validated.input_was_sorted,
            "input_had_flux_error": validated.flux_error is not None,
            "false_alarm_enabled": fap_threshold is not None,
            "false_alarm_probability_threshold": fap_threshold,
            "false_alarm_method": result_fap_method,
            "nterms": self.nterms,
        }
        return LombScargleResult(
            frequency=frequency_array,
            power=power_array,
            dominant_frequency=dominant_frequency,
            dominant_period=float(1.0 / dominant_frequency),
            dominant_power=dominant_power,
            false_alarm_probability=fap_value,
            false_alarm_power_level=fap_power_level,
            false_alarm_probability_threshold=fap_threshold,
            is_significant=significant,
            false_alarm_method=result_fap_method,
            minimum_frequency=float(frequency_array[0]),
            maximum_frequency=float(frequency_array[-1]),
            frequency_step=frequency_step,
            frequency_samples=int(frequency_array.size),
            samples_per_peak=spp,
            normalization=norm,
            fit_mean=self.fit_mean,
            center_data=self.center_data,
            nterms=self.nterms,
            computation_method_requested=method,
            computation_method_used=method_used,
            n_observations=validated.n_observations,
            n_distinct_times=validated.n_distinct_times,
            baseline_days=validated.baseline_days,
            median_cadence_days=validated.median_cadence_days,
            flux_representation=representation,
            used_flux_error=validated.flux_error is not None,
            runtime_seconds=runtime_seconds,
            warnings=tuple(messages),
            metadata=metadata,
        )

    def compute_power_on_grid(
        self,
        time: ArrayLike,
        flux: ArrayLike,
        frequency: ArrayLike,
        flux_error: ArrayLike | None = None,
        *,
        normalization: Normalization | None = None,
        computation_method: ComputationMethod | None = None,
    ) -> NDArray[np.float64]:
        """Return a new, writable float64 power array on an explicit grid."""
        norm = (
            self.normalization
            if normalization is None
            else self._validate_normalization(normalization)
        )
        method = (
            self.computation_method
            if computation_method is None
            else self._validate_computation_method(computation_method)
        )
        validated = self._validate_input(time, flux, flux_error)
        frequency_array = self._validate_frequency_grid(frequency)
        model = self._build_model(validated, norm)
        try:
            power = model.power(frequency_array, method=method)
        except Exception as exc:
            raise RuntimeError(
                f"Astropy could not compute power on the explicit grid with method {method!r}."
            ) from exc
        power_array = np.array(power, dtype=np.float64, copy=True)
        if power_array.ndim != 1 or power_array.size != frequency_array.size:
            raise ValueError("Astropy power must be 1D and match frequency length.")
        if not np.all(np.isfinite(power_array)):
            raise ValueError("power must contain only finite values.")
        return power_array

    def compute_periodogram(
        self,
        time: ArrayLike,
        flux: ArrayLike,
        flux_error: ArrayLike | None = None,
    ) -> LombScargleResult:
        """Return the standard public periodogram without false-alarm work.

        This compatibility surface is used by the public spectral-stability
        component and delegates to :meth:`analyze` without changing its
        frequency-grid or normalization semantics.
        """
        return self.analyze(
            time,
            flux,
            flux_error,
            false_alarm_probability=None,
        )

    @staticmethod
    def dominant_frequency(
        frequency: ArrayLike,
        power: ArrayLike,
    ) -> float:
        """Return the frequency at the first maximum-power grid point."""
        frequency_array = np.asarray(frequency, dtype=np.float64)
        power_array = np.asarray(power, dtype=np.float64)
        if (
            frequency_array.ndim != 1
            or power_array.ndim != 1
            or frequency_array.size == 0
            or frequency_array.size != power_array.size
            or not np.all(np.isfinite(frequency_array))
            or not np.all(np.isfinite(power_array))
        ):
            raise ValueError("frequency and power must be equal finite 1D arrays")
        return float(frequency_array[int(np.argmax(power_array))])

    @staticmethod
    def maximum_power(power: ArrayLike) -> float:
        """Return the finite maximum periodogram power."""
        power_array = np.asarray(power, dtype=np.float64)
        if (
            power_array.ndim != 1
            or power_array.size == 0
            or not np.all(np.isfinite(power_array))
        ):
            raise ValueError("power must be a non-empty finite 1D array")
        return float(np.max(power_array))

    def _validate_input(
        self, time: ArrayLike, flux: ArrayLike, flux_error: ArrayLike | None
    ) -> _ValidatedInput:
        time_array = np.asarray(time, dtype=np.float64)
        flux_array = np.asarray(flux, dtype=np.float64)
        if time_array.ndim != 1 or flux_array.ndim != 1:
            raise ValueError("time and flux must be 1D arrays.")
        if time_array.size != flux_array.size:
            raise ValueError("time and flux must have the same length.")
        if time_array.size < self.minimum_observations:
            raise ValueError(f"At least {self.minimum_observations} observations are required.")
        if not np.all(np.isfinite(time_array)):
            raise ValueError("time must contain only finite values.")
        if not np.all(np.isfinite(flux_array)):
            raise ValueError("flux must contain only finite values.")
        if np.ptp(flux_array) == 0:
            raise ValueError("flux must not be constant.")
        error_array = None
        if flux_error is not None:
            error_array = np.asarray(flux_error, dtype=np.float64)
            if error_array.ndim != 1 or error_array.size != time_array.size:
                raise ValueError("flux_error must be 1D and match time length.")
            if not np.all(np.isfinite(error_array)):
                raise ValueError("flux_error must contain only finite values.")
            if np.any(error_array <= 0):
                raise ValueError("flux_error values must be strictly positive.")

        input_was_sorted = bool(np.all(np.diff(time_array) > 0))
        order = np.argsort(time_array, kind="stable")
        sorted_time = np.array(time_array[order], dtype=np.float64, copy=True)
        differences = np.diff(sorted_time)
        if np.any(differences == 0):
            raise ValueError(
                "Duplicate observation times are not allowed; aggregate duplicates in the loader."
            )
        if np.any(differences <= 0):
            raise ValueError("time must be strictly increasing after stable sorting.")
        sorted_flux = np.array(flux_array[order], dtype=np.float64, copy=True)
        sorted_error = (
            None
            if error_array is None
            else np.array(error_array[order], dtype=np.float64, copy=True)
        )
        return _ValidatedInput(
            time=sorted_time,
            flux=sorted_flux,
            flux_error=sorted_error,
            input_was_sorted=input_was_sorted,
            n_observations=int(sorted_time.size),
            n_distinct_times=int(np.unique(sorted_time).size),
            baseline_days=float(sorted_time[-1] - sorted_time[0]),
            median_cadence_days=float(np.median(differences)),
        )

    def _build_model(self, data: _ValidatedInput, normalization: str) -> LombScargle:
        return LombScargle(
            data.time,
            data.flux,
            dy=data.flux_error,
            fit_mean=self.fit_mean,
            center_data=self.center_data,
            nterms=self.nterms,
            normalization=normalization,
        )

    def _compute_false_alarm(
        self,
        model: LombScargle,
        power: float,
        probability: float,
        method: str,
        fmin: float,
        fmax: float,
        spp: int,
    ) -> tuple[float, float]:
        try:
            fap = model.false_alarm_probability(
                power,
                method=method,
                minimum_frequency=fmin,
                maximum_frequency=fmax,
                samples_per_peak=spp,
            )
            level = model.false_alarm_level(
                probability,
                method=method,
                minimum_frequency=fmin,
                maximum_frequency=fmax,
                samples_per_peak=spp,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Astropy false-alarm calculation failed with method {method!r}, "
                f"normalization {model.normalization!r}, and nterms={self.nterms}."
            ) from exc
        fap_value, level_value = float(fap), float(level)
        if not np.isfinite(fap_value) or not 0 <= fap_value <= 1:
            raise RuntimeError("Astropy returned an invalid false-alarm probability.")
        if not np.isfinite(level_value):
            raise RuntimeError("Astropy returned an invalid false-alarm power level.")
        return fap_value, level_value

    def _resolve_options(
        self,
        minimum_frequency: float | None,
        maximum_frequency: float | None,
        samples_per_peak: int | None,
        normalization: str | None,
        computation_method: str | None,
        false_alarm_probability: object,
    ) -> tuple[float, float, int, str, str, float | None]:
        fmin = (
            self.minimum_frequency
            if minimum_frequency is None
            else self._validate_frequency_bound(
                minimum_frequency,
                "minimum_frequency",
            )
        )
        fmax = (
            self.maximum_frequency
            if maximum_frequency is None
            else self._validate_frequency_bound(
                maximum_frequency,
                "maximum_frequency",
            )
        )
        if fmax <= fmin:
            raise ValueError("maximum_frequency must be greater than minimum_frequency.")
        spp = (
            self.samples_per_peak
            if samples_per_peak is None
            else self._validate_positive_integer(
                samples_per_peak,
                "samples_per_peak",
            )
        )
        norm = (
            self.normalization
            if normalization is None
            else self._validate_normalization(normalization)
        )
        method = (
            self.computation_method
            if computation_method is None
            else self._validate_computation_method(computation_method)
        )
        threshold = (
            self.false_alarm_probability
            if false_alarm_probability is _USE_ANALYZER_DEFAULT
            else self._validate_probability(false_alarm_probability)
        )
        return fmin, fmax, spp, norm, method, threshold

    @staticmethod
    def _validate_frequency_bound(value: object, name: str) -> float:
        result = float(value)
        if not np.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be finite and greater than zero.")
        return result

    @staticmethod
    def _validate_positive_integer(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer.")
        result = int(value)
        if result <= 0:
            raise ValueError(f"{name} must be positive.")
        return result

    @staticmethod
    def _validate_probability(value: object) -> float | None:
        if value is None:
            return None
        result = float(value)
        if not np.isfinite(result) or not 0 < result < 1:
            raise ValueError(
                "false_alarm_probability must be None or strictly between 0 and 1."
            )
        return result

    def _validate_normalization(self, value: str) -> str:
        if value not in self._VALID_NORMALIZATIONS:
            raise ValueError("normalization is not supported.")
        return value

    def _validate_computation_method(self, value: str) -> str:
        if value not in self._VALID_COMPUTATION_METHODS:
            raise ValueError("computation_method is not supported.")
        return value

    def _validate_false_alarm_method(self, value: str) -> str:
        if value not in self._VALID_FALSE_ALARM_METHODS:
            raise ValueError("false_alarm_method is not supported.")
        return value

    def _validate_flux_representation(self, value: str) -> str:
        if value not in self._VALID_FLUX_REPRESENTATIONS:
            raise ValueError("flux_representation is not supported.")
        return value

    @staticmethod
    def _validate_frequency_grid(frequency: ArrayLike) -> NDArray[np.float64]:
        result = np.array(frequency, dtype=np.float64, copy=True)
        if result.ndim != 1 or result.size == 0:
            raise ValueError("frequency must be a non-empty 1D array.")
        if not np.all(np.isfinite(result)):
            raise ValueError("frequency must contain only finite values.")
        if np.any(result <= 0):
            raise ValueError("frequency must be strictly positive.")
        if result.size > 1 and np.any(np.diff(result) <= 0):
            raise ValueError("frequency must be strictly increasing without duplicates.")
        return result

    @staticmethod
    def _validate_periodogram(
        frequency: ArrayLike, power: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        frequency_array = np.array(frequency, dtype=np.float64, copy=True)
        power_array = np.array(power, dtype=np.float64, copy=True)
        if frequency_array.ndim != 1 or power_array.ndim != 1:
            raise ValueError("frequency and power must be 1D arrays.")
        if frequency_array.size < 2 or power_array.size != frequency_array.size:
            raise ValueError("periodogram arrays must have equal length of at least 2.")
        if not np.all(np.isfinite(frequency_array)) or not np.all(np.isfinite(power_array)):
            raise ValueError("periodogram arrays must contain only finite values.")
        if np.any(frequency_array <= 0) or np.any(np.diff(frequency_array) <= 0):
            raise ValueError("periodogram frequency must be positive and strictly increasing.")
        return frequency_array, power_array

    def _grid_diagnostics(self, frequency: NDArray[np.float64]) -> tuple[float, bool]:
        differences = np.diff(frequency)
        step = float(np.median(differences))
        uniform = bool(
            np.allclose(
                differences,
                step,
                rtol=self._GRID_UNIFORMITY_RTOL,
                atol=self._GRID_UNIFORMITY_ATOL,
            )
        )
        return step, uniform

    def _append_large_grid_warning(self, messages: list[str], size: int) -> None:
        if size > self._LARGE_GRID_WARNING_THRESHOLD:
            messages.append(
                f"The frequency grid contains {size} samples, exceeding the "
                f"conservative warning threshold of {self._LARGE_GRID_WARNING_THRESHOLD}."
            )
