"""Spectral stability features for Physics-Informed Spectral Descriptor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stellar_anomaly_detection.pisd.policy import (
    DOMINANT_PEAK_RAYLEIGH_FACTOR,
    STABILITY_EPSILON,
    STABILITY_FALLBACK_NOISE_FRACTION,
    STABILITY_NEGATIVE_POWER_TOLERANCE,
    STABILITY_NOISE_SCALE_FACTOR,
    STABILITY_N_PERTURBATIONS,
    STABILITY_RANDOM_STATE,
)
from stellar_anomaly_detection.spectral.lomb_scargle import LombScargleAnalyzer


@dataclass(frozen=True)
class NoiseScaleResult:
    """Noise scale used to perturb an observed light curve.

    Parameters
    ----------
    noise_scale : np.ndarray
        Per-observation standard deviations used for flux perturbations.
    noise_source : str
        Source used to estimate the noise scale.
    noise_scale_median : float
        Median of the per-observation noise scale.
    noise_scale_mean : float
        Mean of the per-observation noise scale.
    noise_scale_minimum : float
        Minimum per-observation noise scale.
    noise_scale_maximum : float
        Maximum per-observation noise scale.
    """

    noise_scale: NDArray[np.float64]
    noise_source: str
    noise_scale_median: float
    noise_scale_mean: float
    noise_scale_minimum: float
    noise_scale_maximum: float

    def to_dict(self) -> dict[str, NDArray[np.float64] | float | str]:
        """Return a dictionary representation while preserving arrays."""
        return {
            "noise_scale": self.noise_scale,
            "noise_source": self.noise_source,
            "noise_scale_median": self.noise_scale_median,
            "noise_scale_mean": self.noise_scale_mean,
            "noise_scale_minimum": self.noise_scale_minimum,
            "noise_scale_maximum": self.noise_scale_maximum,
        }


@dataclass(frozen=True)
class SpectralStabilityResult:
    """Detailed result of Monte Carlo spectral stability analysis.

    Parameters
    ----------
    reference_frequency : np.ndarray
        Frequency grid of the reference periodogram.
    reference_power : np.ndarray
        Power of the reference periodogram.
    reference_dominant_frequency : float
        Frequency of the reference global maximum.
    reference_maximum_power : float
        Power at the reference global maximum.
    profile_similarities : np.ndarray
        Jensen--Shannon similarities for all perturbations.
    perturbed_dominant_frequencies : np.ndarray
        Dominant frequencies for all perturbations.
    normalized_frequency_drifts : np.ndarray
        Frequency drifts measured in Rayleigh resolution units.
    dominant_power_log_changes : np.ndarray
        Absolute logarithmic changes in dominant power.
    noise_source : str
        Source used to estimate the perturbation scale.
    noise_scale_median, noise_scale_mean : float
        Central summaries of the perturbation scale.
    noise_scale_minimum, noise_scale_maximum : float
        Range of the perturbation scale.
    perturbation_count : int
        Number of Monte Carlo realizations.
    time_baseline : float
        Observation time baseline.
    rayleigh_resolution : float
        Inverse observation time baseline.
    random_state : int
        Seed used for the local random generator.
    """

    reference_frequency: NDArray[np.float64]
    reference_power: NDArray[np.float64]
    reference_dominant_frequency: float
    reference_maximum_power: float
    profile_similarities: NDArray[np.float64]
    perturbed_dominant_frequencies: NDArray[np.float64]
    normalized_frequency_drifts: NDArray[np.float64]
    dominant_power_log_changes: NDArray[np.float64]
    noise_source: str
    noise_scale_median: float
    noise_scale_mean: float
    noise_scale_minimum: float
    noise_scale_maximum: float
    perturbation_count: int
    time_baseline: float
    rayleigh_resolution: float
    random_state: int

    def to_dict(
        self,
    ) -> dict[str, NDArray[np.float64] | float | int | str]:
        """Return a dictionary representation while preserving arrays."""
        return {
            "reference_frequency": self.reference_frequency,
            "reference_power": self.reference_power,
            "reference_dominant_frequency": self.reference_dominant_frequency,
            "reference_maximum_power": self.reference_maximum_power,
            "profile_similarities": self.profile_similarities,
            "perturbed_dominant_frequencies": (
                self.perturbed_dominant_frequencies
            ),
            "normalized_frequency_drifts": self.normalized_frequency_drifts,
            "dominant_power_log_changes": self.dominant_power_log_changes,
            "noise_source": self.noise_source,
            "noise_scale_median": self.noise_scale_median,
            "noise_scale_mean": self.noise_scale_mean,
            "noise_scale_minimum": self.noise_scale_minimum,
            "noise_scale_maximum": self.noise_scale_maximum,
            "perturbation_count": self.perturbation_count,
            "time_baseline": self.time_baseline,
            "rayleigh_resolution": self.rayleigh_resolution,
            "random_state": self.random_state,
        }


class SpectralStability:
    """Measure periodogram robustness to plausible flux perturbations."""

    _BOUND_TOLERANCE = 1e-10
    _NOISE_SOURCES = frozenset(
        {"flux_error", "first_difference", "robust_flux_fallback"}
    )

    def __init__(
        self,
        analyzer: LombScargleAnalyzer,
        n_perturbations: int = STABILITY_N_PERTURBATIONS,
        random_state: int = STABILITY_RANDOM_STATE,
        noise_scale_factor: float = STABILITY_NOISE_SCALE_FACTOR,
        fallback_noise_fraction: float = STABILITY_FALLBACK_NOISE_FRACTION,
        negative_power_tolerance: float = (
            STABILITY_NEGATIVE_POWER_TOLERANCE
        ),
        epsilon: float = STABILITY_EPSILON,
        dominant_peak_rayleigh_factor: float = (
            DOMINANT_PEAK_RAYLEIGH_FACTOR
        ),
    ) -> None:
        self.analyzer = analyzer
        self.n_perturbations = n_perturbations
        self.random_state = random_state
        self.noise_scale_factor = float(noise_scale_factor)
        self.fallback_noise_fraction = float(fallback_noise_fraction)
        self.negative_power_tolerance = float(negative_power_tolerance)
        self.epsilon = float(epsilon)
        self.dominant_peak_rayleigh_factor = float(
            dominant_peak_rayleigh_factor
        )
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate spectral stability configuration."""
        if not isinstance(self.analyzer, LombScargleAnalyzer):
            raise TypeError("analyzer must be a LombScargleAnalyzer instance.")

        if isinstance(self.n_perturbations, bool) or not isinstance(
            self.n_perturbations, (int, np.integer)
        ):
            raise TypeError("n_perturbations must be an integer.")
        if self.n_perturbations < 1:
            raise ValueError("n_perturbations must be at least 1.")

        if isinstance(self.random_state, bool) or not isinstance(
            self.random_state, (int, np.integer)
        ):
            raise TypeError("random_state must be an integer.")
        if self.random_state < 0:
            raise ValueError("random_state must be non-negative.")

        self.n_perturbations = int(self.n_perturbations)
        self.random_state = int(self.random_state)

        self._validate_positive_configuration(
            self.noise_scale_factor, "noise_scale_factor"
        )
        self._validate_positive_configuration(
            self.fallback_noise_fraction, "fallback_noise_fraction"
        )
        if not np.isfinite(self.negative_power_tolerance):
            raise ValueError("negative_power_tolerance must be finite.")
        if self.negative_power_tolerance < 0:
            raise ValueError(
                "negative_power_tolerance must be non-negative."
            )
        self._validate_positive_configuration(self.epsilon, "epsilon")
        self._validate_positive_configuration(
            self.dominant_peak_rayleigh_factor,
            "dominant_peak_rayleigh_factor",
        )

    @staticmethod
    def _validate_positive_configuration(value: float, name: str) -> None:
        """Validate a finite, strictly positive configuration scalar."""
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero.")

    def validate_input(
        self,
        time: ArrayLike,
        flux: ArrayLike,
        flux_error: ArrayLike | None = None,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64] | None,
    ]:
        """Validate a light curve using the configured analyzer.

        Parameters
        ----------
        time : array-like
            Observation times.
        flux : array-like
            Observed flux values.
        flux_error : array-like, optional
            Positive observational flux uncertainties.

        Returns
        -------
        tuple of np.ndarray
            Independent copies of validated, time-sorted inputs.
        """
        valid_time, valid_flux, valid_flux_error = self.analyzer.validate_input(
            time, flux, flux_error
        )
        time_baseline = float(np.max(valid_time) - np.min(valid_time))
        if not np.isfinite(time_baseline) or time_baseline <= 0:
            raise ValueError("time baseline must be finite and positive.")

        error_copy = None
        if valid_flux_error is not None:
            error_copy = valid_flux_error.astype(float, copy=True)
        return (
            valid_time.astype(float, copy=True),
            valid_flux.astype(float, copy=True),
            error_copy,
        )

    def estimate_noise_scale(
        self,
        flux: ArrayLike,
        flux_error: ArrayLike | None = None,
    ) -> NoiseScaleResult:
        """Estimate per-observation perturbation standard deviations.

        Parameters
        ----------
        flux : array-like
            Validated or raw one-dimensional flux values.
        flux_error : array-like, optional
            Positive heteroscedastic observational errors.

        Returns
        -------
        NoiseScaleResult
            Noise vector, its provenance, and distribution summaries.
        """
        flux_array = np.asarray(flux, dtype=float)
        self._validate_vector(flux_array, "flux")

        if flux_error is not None:
            error_array = np.asarray(flux_error, dtype=float)
            self._validate_vector(error_array, "flux_error")
            if error_array.size != flux_array.size:
                raise ValueError(
                    "flux_error must have the same length as flux."
                )
            if np.any(error_array <= 0):
                raise ValueError("flux_error must be strictly positive.")
            noise_scale = self.noise_scale_factor * error_array
            noise_source = "flux_error"
        else:
            differences = np.diff(flux_array)
            median_difference = float(np.median(differences))
            mad_difference = float(
                np.median(np.abs(differences - median_difference))
            )
            first_difference_sigma = (
                1.4826 * mad_difference / np.sqrt(2.0)
            )

            if first_difference_sigma > self.epsilon:
                noise_scale = (
                    self.noise_scale_factor
                    * first_difference_sigma
                    * np.ones_like(flux_array)
                )
                noise_source = "first_difference"
            else:
                median_flux = float(np.median(flux_array))
                mad_flux = float(
                    np.median(np.abs(flux_array - median_flux))
                )
                robust_flux_scale = 1.4826 * mad_flux
                fallback_sigma = (
                    self.fallback_noise_fraction * robust_flux_scale
                )
                if fallback_sigma <= self.epsilon:
                    raise ValueError(
                        "robust fallback noise scale must be greater than "
                        "epsilon."
                    )
                noise_scale = (
                    self.noise_scale_factor
                    * fallback_sigma
                    * np.ones_like(flux_array)
                )
                noise_source = "robust_flux_fallback"

        noise_scale = np.asarray(noise_scale, dtype=float)
        self._validate_vector(noise_scale, "noise_scale")
        if noise_scale.size != flux_array.size:
            raise ValueError("noise_scale must have the same length as flux.")
        if np.any(noise_scale <= 0):
            raise ValueError("noise_scale must be strictly positive.")
        if noise_source not in self._NOISE_SOURCES:
            raise RuntimeError("noise_source is invalid.")

        return NoiseScaleResult(
            noise_scale=noise_scale.astype(float, copy=True),
            noise_source=noise_source,
            noise_scale_median=self._finite_float(
                np.median(noise_scale), "noise_scale_median"
            ),
            noise_scale_mean=self._finite_float(
                np.mean(noise_scale), "noise_scale_mean"
            ),
            noise_scale_minimum=self._finite_float(
                np.min(noise_scale), "noise_scale_minimum"
            ),
            noise_scale_maximum=self._finite_float(
                np.max(noise_scale), "noise_scale_maximum"
            ),
        )

    def generate_perturbed_flux(
        self,
        flux: NDArray[np.float64],
        noise_scale: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Generate one Gaussian flux perturbation without mutating inputs."""
        flux_array = np.asarray(flux, dtype=float)
        noise_array = np.asarray(noise_scale, dtype=float)
        self._validate_vector(flux_array, "flux")
        self._validate_vector(noise_array, "noise_scale")
        if flux_array.size != noise_array.size:
            raise ValueError("noise_scale must have the same length as flux.")
        if np.any(noise_array <= 0):
            raise ValueError("noise_scale must be strictly positive.")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator.")

        perturbed_flux = flux_array + rng.normal(
            loc=0.0,
            scale=noise_array,
            size=flux_array.size,
        )
        self._validate_vector(perturbed_flux, "perturbed_flux")
        if perturbed_flux.size != flux_array.size:
            raise RuntimeError(
                "perturbed_flux must have the same length as flux."
            )
        return perturbed_flux.astype(float, copy=True)

    def normalize_power_distribution(
        self,
        power: ArrayLike,
    ) -> NDArray[np.float64]:
        """Convert periodogram power into a discrete probability vector."""
        power_array = np.asarray(power, dtype=float)
        self._validate_vector(power_array, "power")
        if np.any(power_array < -self.negative_power_tolerance):
            raise ValueError(
                "power contains values below the allowed negative tolerance."
            )

        nonnegative_power = power_array.astype(float, copy=True)
        nonnegative_power[nonnegative_power < 0] = 0.0
        total_power = self._finite_float(
            np.sum(nonnegative_power), "total_power"
        )
        if total_power <= self.epsilon:
            raise ValueError("total spectral power must exceed epsilon.")

        distribution = nonnegative_power / total_power
        if not np.all(np.isfinite(distribution)):
            raise RuntimeError("power distribution must be finite.")
        if np.any(distribution < 0):
            raise RuntimeError("power distribution must be non-negative.")
        if not np.isclose(np.sum(distribution), 1.0):
            raise RuntimeError("power distribution must sum approximately to 1.")
        return distribution.astype(float, copy=True)

    def jensen_shannon_similarity(
        self,
        reference_distribution: ArrayLike,
        perturbed_distribution: ArrayLike,
    ) -> float:
        """Compute normalized Jensen--Shannon profile similarity."""
        reference = self._validate_distribution(
            reference_distribution, "reference_distribution"
        )
        perturbed = self._validate_distribution(
            perturbed_distribution, "perturbed_distribution"
        )
        if reference.size != perturbed.size:
            raise ValueError("spectral distributions must have equal length.")

        mixture = 0.5 * (reference + perturbed)
        reference_mask = reference > 0
        perturbed_mask = perturbed > 0
        reference_kl = np.sum(
            reference[reference_mask]
            * np.log(reference[reference_mask] / mixture[reference_mask])
        )
        perturbed_kl = np.sum(
            perturbed[perturbed_mask]
            * np.log(perturbed[perturbed_mask] / mixture[perturbed_mask])
        )
        divergence = 0.5 * reference_kl + 0.5 * perturbed_kl
        similarity = 1.0 - divergence / np.log(2.0)
        return self._bounded_float(similarity, "profile_similarity")

    def run_perturbations(
        self,
        time: NDArray[np.float64],
        flux: NDArray[np.float64],
        flux_error: NDArray[np.float64] | None,
        reference_frequency: NDArray[np.float64],
        reference_power: NDArray[np.float64],
        reference_dominant_frequency: float,
        reference_maximum_power: float,
        noise_scale_result: NoiseScaleResult,
        time_baseline: float,
        rng: np.random.Generator,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Run all perturbations on the fixed reference frequency grid."""
        reference_distribution = self.normalize_power_distribution(
            reference_power
        )
        rayleigh_resolution = 1.0 / time_baseline
        profile_similarities = np.empty(self.n_perturbations, dtype=np.float64)
        perturbed_dominant_frequencies = np.empty(
            self.n_perturbations, dtype=np.float64
        )
        normalized_frequency_drifts = np.empty(
            self.n_perturbations, dtype=np.float64
        )
        dominant_power_log_changes = np.empty(
            self.n_perturbations, dtype=np.float64
        )

        anticipated_errors = (
            ValueError,
            RuntimeError,
            FloatingPointError,
            OverflowError,
        )
        for index in range(self.n_perturbations):
            try:
                perturbed_flux = self.generate_perturbed_flux(
                    flux, noise_scale_result.noise_scale, rng
                )
                perturbed_power = self.analyzer.compute_power_on_grid(
                    time=time,
                    flux=perturbed_flux,
                    frequency=reference_frequency,
                    flux_error=flux_error,
                )
                perturbed_distribution = self.normalize_power_distribution(
                    perturbed_power
                )
                profile_similarity = self.jensen_shannon_similarity(
                    reference_distribution, perturbed_distribution
                )

                maximum_index = int(np.argmax(perturbed_power))
                perturbed_dominant_frequency = float(
                    reference_frequency[maximum_index]
                )
                normalized_drift = (
                    abs(
                        perturbed_dominant_frequency
                        - reference_dominant_frequency
                    )
                    / rayleigh_resolution
                )
                perturbed_maximum_power = float(np.max(perturbed_power))
                log_change = abs(
                    np.log(
                        (perturbed_maximum_power + self.epsilon)
                        / (reference_maximum_power + self.epsilon)
                    )
                )

                profile_similarities[index] = self._bounded_float(
                    profile_similarity, "profile_similarity"
                )
                perturbed_dominant_frequencies[index] = self._finite_float(
                    perturbed_dominant_frequency,
                    "perturbed_dominant_frequency",
                )
                normalized_frequency_drifts[index] = (
                    self._nonnegative_finite_float(
                        normalized_drift, "normalized_frequency_drift"
                    )
                )
                dominant_power_log_changes[index] = (
                    self._nonnegative_finite_float(
                        log_change, "dominant_power_log_change"
                    )
                )
            except anticipated_errors as exc:
                raise RuntimeError(
                    f"Spectral stability perturbation {index} failed."
                ) from exc

        outputs = (
            profile_similarities,
            perturbed_dominant_frequencies,
            normalized_frequency_drifts,
            dominant_power_log_changes,
        )
        for output in outputs:
            self._validate_perturbation_output(output)
        return outputs

    def analyze(
        self,
        time: ArrayLike,
        flux: ArrayLike,
        flux_error: ArrayLike | None = None,
    ) -> SpectralStabilityResult:
        """Run reproducible Monte Carlo spectral stability analysis."""
        valid_time, valid_flux, valid_flux_error = self.validate_input(
            time, flux, flux_error
        )
        time_baseline = self._nonnegative_finite_float(
            np.max(valid_time) - np.min(valid_time), "time_baseline"
        )
        if time_baseline <= 0:
            raise ValueError("time baseline must be positive.")
        rayleigh_resolution = self._nonnegative_finite_float(
            1.0 / time_baseline, "rayleigh_resolution"
        )

        noise_scale_result = self.estimate_noise_scale(
            valid_flux, valid_flux_error
        )
        reference_periodogram = self.analyzer.compute_periodogram(
            time=valid_time,
            flux=valid_flux,
            flux_error=valid_flux_error,
        )
        reference_frequency = np.asarray(
            reference_periodogram.frequency, dtype=float
        ).astype(float, copy=True)
        reference_power = np.asarray(
            reference_periodogram.power, dtype=float
        ).astype(float, copy=True)
        reference_dominant_frequency = self._finite_float(
            self.analyzer.dominant_frequency(
                reference_frequency, reference_power
            ),
            "reference_dominant_frequency",
        )
        reference_maximum_power = self._finite_float(
            self.analyzer.maximum_power(reference_power),
            "reference_maximum_power",
        )

        rng = np.random.default_rng(self.random_state)
        (
            profile_similarities,
            perturbed_dominant_frequencies,
            normalized_frequency_drifts,
            dominant_power_log_changes,
        ) = self.run_perturbations(
            time=valid_time,
            flux=valid_flux,
            flux_error=valid_flux_error,
            reference_frequency=reference_frequency,
            reference_power=reference_power,
            reference_dominant_frequency=reference_dominant_frequency,
            reference_maximum_power=reference_maximum_power,
            noise_scale_result=noise_scale_result,
            time_baseline=time_baseline,
            rng=rng,
        )

        return SpectralStabilityResult(
            reference_frequency=reference_frequency,
            reference_power=reference_power,
            reference_dominant_frequency=reference_dominant_frequency,
            reference_maximum_power=reference_maximum_power,
            profile_similarities=profile_similarities,
            perturbed_dominant_frequencies=perturbed_dominant_frequencies,
            normalized_frequency_drifts=normalized_frequency_drifts,
            dominant_power_log_changes=dominant_power_log_changes,
            noise_source=noise_scale_result.noise_source,
            noise_scale_median=noise_scale_result.noise_scale_median,
            noise_scale_mean=noise_scale_result.noise_scale_mean,
            noise_scale_minimum=noise_scale_result.noise_scale_minimum,
            noise_scale_maximum=noise_scale_result.noise_scale_maximum,
            perturbation_count=self.n_perturbations,
            time_baseline=time_baseline,
            rayleigh_resolution=rayleigh_resolution,
            random_state=self.random_state,
        )

    def median_profile_similarity(
        self, result: SpectralStabilityResult
    ) -> float:
        """Return the median Jensen--Shannon similarity."""
        return self._bounded_float(
            np.median(result.profile_similarities),
            "median_profile_similarity",
        )

    def profile_similarity_variability(
        self, result: SpectralStabilityResult
    ) -> float:
        """Return population variability of profile similarity."""
        return self._nonnegative_finite_float(
            np.std(result.profile_similarities, ddof=0),
            "profile_similarity_variability",
        )

    def dominant_peak_retention_rate(
        self, result: SpectralStabilityResult
    ) -> float:
        """Return the fraction of peaks retained within the set tolerance."""
        retention = np.mean(
            result.normalized_frequency_drifts
            <= self.dominant_peak_rayleigh_factor
        )
        return self._bounded_float(retention, "dominant_peak_retention_rate")

    def median_normalized_frequency_drift(
        self, result: SpectralStabilityResult
    ) -> float:
        """Return the median dominant-frequency drift in Rayleigh units."""
        return self._nonnegative_finite_float(
            np.median(result.normalized_frequency_drifts),
            "median_normalized_frequency_drift",
        )

    def p90_normalized_frequency_drift(
        self, result: SpectralStabilityResult
    ) -> float:
        """Return the 90th percentile frequency drift in Rayleigh units."""
        return self._nonnegative_finite_float(
            np.percentile(result.normalized_frequency_drifts, 90),
            "p90_normalized_frequency_drift",
        )

    def dominant_power_stability(
        self, result: SpectralStabilityResult
    ) -> float:
        """Return exponential stability of median dominant-power change."""
        stability = np.exp(-np.median(result.dominant_power_log_changes))
        return self._bounded_float(stability, "dominant_power_stability")

    def dominant_power_log_variability(
        self, result: SpectralStabilityResult
    ) -> float:
        """Return population variability of dominant-power log changes."""
        return self._nonnegative_finite_float(
            np.std(result.dominant_power_log_changes, ddof=0),
            "dominant_power_log_variability",
        )

    def extract(
        self,
        time: ArrayLike,
        flux: ArrayLike,
        flux_error: ArrayLike | None = None,
    ) -> dict[str, float]:
        """Extract the seven spectral stability features in fixed order."""
        result = self.analyze(time, flux, flux_error)
        features = {
            "median_profile_similarity": self.median_profile_similarity(result),
            "profile_similarity_variability": (
                self.profile_similarity_variability(result)
            ),
            "dominant_peak_retention_rate": (
                self.dominant_peak_retention_rate(result)
            ),
            "median_normalized_frequency_drift": (
                self.median_normalized_frequency_drift(result)
            ),
            "p90_normalized_frequency_drift": (
                self.p90_normalized_frequency_drift(result)
            ),
            "dominant_power_stability": self.dominant_power_stability(result),
            "dominant_power_log_variability": (
                self.dominant_power_log_variability(result)
            ),
        }
        for name, value in features.items():
            self._finite_float(value, name)
        return features

    @staticmethod
    def _validate_vector(array: NDArray[np.float64], name: str) -> None:
        """Validate a finite, nonempty, one-dimensional array."""
        if array.ndim != 1:
            raise ValueError(f"{name} must be a 1D array.")
        if array.size == 0:
            raise ValueError(f"{name} must not be empty.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")

    def _validate_distribution(
        self, distribution: ArrayLike, name: str
    ) -> NDArray[np.float64]:
        """Validate and copy a discrete probability distribution."""
        array = np.asarray(distribution, dtype=float)
        self._validate_vector(array, name)
        if np.any(array < 0):
            raise ValueError(f"{name} must be non-negative.")
        if not np.isclose(np.sum(array), 1.0):
            raise ValueError(f"{name} must sum approximately to 1.")
        return array.astype(float, copy=True)

    def _validate_perturbation_output(
        self, output: NDArray[np.float64]
    ) -> None:
        """Validate a completed per-realization metric array."""
        if output.ndim != 1 or output.size != self.n_perturbations:
            raise RuntimeError(
                "perturbation output must be 1D with n_perturbations values."
            )
        if output.dtype != np.dtype(np.float64):
            raise RuntimeError("perturbation output must have float64 dtype.")
        if not np.all(np.isfinite(output)):
            raise RuntimeError("perturbation output must contain finite values.")

    @staticmethod
    def _finite_float(value: float, name: str) -> float:
        """Return a finite Python float."""
        scalar = float(value)
        if not np.isfinite(scalar):
            raise ValueError(f"{name} must be finite.")
        return scalar

    def _nonnegative_finite_float(self, value: float, name: str) -> float:
        """Return a finite, nonnegative Python float."""
        scalar = self._finite_float(value, name)
        if scalar < 0:
            raise RuntimeError(f"{name} must be non-negative.")
        return scalar

    def _bounded_float(self, value: float, name: str) -> float:
        """Bound a scalar to [0, 1], tolerating only numerical error."""
        scalar = self._finite_float(value, name)
        if scalar < -self._BOUND_TOLERANCE or scalar > 1 + self._BOUND_TOLERANCE:
            raise RuntimeError(f"{name} is outside the theoretical [0, 1] range.")
        if scalar < 0:
            return 0.0
        if scalar > 1:
            return 1.0
        return scalar
