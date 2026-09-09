"""Spectral complexity features for Physics-Informed Spectral Descriptor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stellar_anomaly_detection.constants import (
    POWER_NEGATIVE_TOLERANCE,
    SPECTRAL_COMPLEXITY_EPSILON,
    TOP_POWER_FRACTION,
)


@dataclass(frozen=True)
class PowerDistributionResult:
    """
    Validated spectral power distribution.

    Parameters
    frequency : np.ndarray
        Validated frequency grid.
    nonnegative_power : np.ndarray
        Power values after correcting tiny numerical negatives.
    normalized_power : np.ndarray
        Discrete power distribution with sum equal to 1.
    total_power : float
        Sum of nonnegative power.
    mean_power : float
        Mean nonnegative power.
    standard_deviation : float
        Population standard deviation of nonnegative power.
    median_power : float
        Median nonnegative power.
    mad_power : float
        Median absolute deviation of nonnegative power.
    point_count : int
        Number of spectral points.
    """

    frequency: NDArray[np.float64]
    nonnegative_power: NDArray[np.float64]
    normalized_power: NDArray[np.float64]
    total_power: float
    mean_power: float
    standard_deviation: float
    median_power: float
    mad_power: float
    point_count: int

    def to_dict(
        self,
    ) -> dict[str, NDArray[np.float64] | float | int]:
        """
        Return the distribution result as a dictionary.

        Returns
        dict
            Dictionary representation. NumPy arrays are preserved.
        """
        return {
            "frequency": self.frequency,
            "nonnegative_power": self.nonnegative_power,
            "normalized_power": self.normalized_power,
            "total_power": self.total_power,
            "mean_power": self.mean_power,
            "standard_deviation": self.standard_deviation,
            "median_power": self.median_power,
            "mad_power": self.mad_power,
            "point_count": self.point_count,
        }


class SpectralComplexity:
    """
    Extract global spectral power complexity features.

    Notes
    The extracted features assume that all periodograms are computed using
    consistent frequency units, frequency bounds, grid resolution and
    Lomb-Scargle normalization. The discrete power-distribution measures also
    assume a common or approximately uniform frequency grid.
    """

    def __init__(
        self,
        epsilon: float = SPECTRAL_COMPLEXITY_EPSILON,
        negative_tolerance: float = POWER_NEGATIVE_TOLERANCE,
        top_power_fraction: float = TOP_POWER_FRACTION,
    ) -> None:
        self.epsilon = float(epsilon)
        self.negative_tolerance = float(negative_tolerance)
        self.top_power_fraction = float(top_power_fraction)

        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate spectral complexity configuration."""
        if not np.isfinite(self.epsilon):
            raise ValueError("epsilon must be finite.")

        if self.epsilon <= 0:
            raise ValueError("epsilon must be greater than zero.")

        if not np.isfinite(self.negative_tolerance):
            raise ValueError("negative_tolerance must be finite.")

        if self.negative_tolerance < 0:
            raise ValueError("negative_tolerance must be non-negative.")

        if not np.isfinite(self.top_power_fraction):
            raise ValueError("top_power_fraction must be finite.")

        if not 0 < self.top_power_fraction < 1:
            raise ValueError(
                "top_power_fraction must be strictly between 0 and 1."
            )

    def validate_input(
        self,
        frequency: ArrayLike,
        power: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Validate periodogram frequency and power arrays."""
        frequency_array = np.asarray(frequency, dtype=float)
        power_array = np.asarray(power, dtype=float)

        if frequency_array.ndim != 1 or power_array.ndim != 1:
            raise ValueError("frequency and power must be 1D arrays.")

        if frequency_array.size == 0 or power_array.size == 0:
            raise ValueError("frequency and power must not be empty.")

        if frequency_array.size != power_array.size:
            raise ValueError("frequency and power must have the same length.")

        if frequency_array.size < 3:
            raise ValueError("At least three spectral points are required.")

        if not np.all(np.isfinite(frequency_array)):
            raise ValueError("frequency must contain only finite values.")

        if not np.all(np.isfinite(power_array)):
            raise ValueError("power must contain only finite values.")

        if np.any(frequency_array <= 0):
            raise ValueError("frequency values must be strictly positive.")

        if np.any(np.diff(frequency_array) <= 0):
            raise ValueError("frequency values must be strictly increasing.")

        return (
            frequency_array.astype(float, copy=True),
            power_array.astype(float, copy=True),
        )

    def prepare_nonnegative_power(
        self,
        power: ArrayLike,
    ) -> NDArray[np.float64]:
        """
        Correct tiny numerical negative power values.

        Parameters
        power : array-like
            Periodogram power values.

        Returns
        np.ndarray
            Nonnegative copy of ``power``.

        Raises
        ValueError
            If power contains significant negative values.
        """
        power_array = np.asarray(power, dtype=float)

        if power_array.ndim != 1:
            raise ValueError("power must be a 1D array.")

        if power_array.size == 0:
            raise ValueError("power must not be empty.")

        if not np.all(np.isfinite(power_array)):
            raise ValueError("power must contain only finite values.")

        if np.any(power_array < -self.negative_tolerance):
            raise ValueError(
                "power contains values below the allowed negative tolerance."
            )

        nonnegative_power = power_array.astype(float, copy=True)
        numerical_negative_mask = (
            (nonnegative_power < 0)
            & (nonnegative_power >= -self.negative_tolerance)
        )
        nonnegative_power[numerical_negative_mask] = 0.0

        if np.any(nonnegative_power < 0):
            raise RuntimeError("nonnegative_power must be non-negative.")

        return nonnegative_power

    def build_power_distribution(
        self,
        frequency: ArrayLike,
        power: ArrayLike,
    ) -> PowerDistributionResult:
        """Build a normalized spectral power distribution."""
        frequency_array, power_array = self.validate_input(frequency, power)
        nonnegative_power = self.prepare_nonnegative_power(power_array)

        total_power = self._finite_float(
            np.sum(nonnegative_power),
            "total_power",
        )
        if total_power <= self.epsilon:
            raise ValueError(
                "total spectral power must be greater than epsilon."
            )

        normalized_power = nonnegative_power / total_power

        if not np.all(np.isfinite(normalized_power)):
            raise RuntimeError("normalized_power must contain finite values.")

        if np.any(normalized_power < 0):
            raise RuntimeError("normalized_power must be non-negative.")

        if not np.isclose(np.sum(normalized_power), 1.0):
            raise RuntimeError("normalized_power must sum approximately to 1.")

        mean_power = self._finite_float(np.mean(nonnegative_power), "mean_power")
        standard_deviation = self._finite_float(
            np.std(nonnegative_power, ddof=0),
            "standard_deviation",
        )
        median_power = self._finite_float(
            np.median(nonnegative_power),
            "median_power",
        )
        mad_power = self._finite_float(
            np.median(np.abs(nonnegative_power - median_power)),
            "mad_power",
        )

        return PowerDistributionResult(
            frequency=frequency_array,
            nonnegative_power=nonnegative_power,
            normalized_power=normalized_power.astype(float, copy=True),
            total_power=total_power,
            mean_power=mean_power,
            standard_deviation=standard_deviation,
            median_power=median_power,
            mad_power=mad_power,
            point_count=int(nonnegative_power.size),
        )

    def normalized_spectral_entropy(
        self,
        distribution: PowerDistributionResult,
    ) -> float:
        """Compute normalized Shannon spectral entropy."""
        self._validate_distribution(distribution)

        positive_power = distribution.normalized_power[
            distribution.normalized_power > 0
        ]
        entropy = -np.sum(positive_power * np.log(positive_power))
        normalized_entropy = entropy / np.log(distribution.point_count)

        return self._bounded_float(
            normalized_entropy,
            "normalized_spectral_entropy",
        )

    def spectral_gini_index(
        self,
        distribution: PowerDistributionResult,
    ) -> float:
        """Compute the spectral Gini index from nonnegative power."""
        self._validate_distribution(distribution)

        sorted_power = np.sort(distribution.nonnegative_power)
        indices = np.arange(1, distribution.point_count + 1, dtype=float)

        gini = (
            2.0 * np.sum(indices * sorted_power)
            / (distribution.point_count * distribution.total_power)
        ) - ((distribution.point_count + 1) / distribution.point_count)

        return self._bounded_float(gini, "spectral_gini_index")

    def normalized_power_variance(
        self,
        distribution: PowerDistributionResult,
    ) -> float:
        """Compute normalized variance of the discrete power distribution."""
        self._validate_distribution(distribution)

        point_count = distribution.point_count
        uniform_probability = 1.0 / point_count
        variance = (
            point_count
            / (point_count - 1)
            * np.sum(
                (distribution.normalized_power - uniform_probability) ** 2
            )
        )

        return self._bounded_float(variance, "normalized_power_variance")

    def power_skewness(
        self,
        distribution: PowerDistributionResult,
    ) -> float:
        """Compute population skewness of nonnegative spectral power."""
        self._validate_distribution(distribution)

        if distribution.standard_deviation <= np.finfo(float).tiny:
            return 0.0

        centered_scaled = (
            distribution.nonnegative_power - distribution.mean_power
        ) / distribution.standard_deviation
        skewness = np.mean(centered_scaled**3)

        return self._finite_float(skewness, "power_skewness")

    def power_excess_kurtosis(
        self,
        distribution: PowerDistributionResult,
    ) -> float:
        """Compute population excess kurtosis of nonnegative spectral power."""
        self._validate_distribution(distribution)

        if distribution.standard_deviation <= np.finfo(float).tiny:
            return 0.0

        centered_scaled = (
            distribution.nonnegative_power - distribution.mean_power
        ) / distribution.standard_deviation
        excess_kurtosis = np.mean(centered_scaled**4) - 3.0

        return self._finite_float(
            excess_kurtosis,
            "power_excess_kurtosis",
        )

    def robust_power_dispersion(
        self,
        distribution: PowerDistributionResult,
    ) -> float:
        """Compute robust MAD-based power dispersion."""
        self._validate_distribution(distribution)

        dispersion = (
            1.4826 * distribution.mad_power / distribution.mean_power
        )

        return self._nonnegative_finite_float(
            dispersion,
            "robust_power_dispersion",
        )

    def top_power_concentration(
        self,
        distribution: PowerDistributionResult,
    ) -> float:
        """Compute normalized concentration in the strongest power bins."""
        self._validate_distribution(distribution)

        point_count = distribution.point_count
        selected_count = max(
            1,
            int(np.ceil(self.top_power_fraction * point_count)),
        )

        sorted_power = np.sort(distribution.normalized_power)[::-1]
        concentration = float(np.sum(sorted_power[:selected_count]))

        uniform_reference = selected_count / point_count
        denominator = 1.0 - uniform_reference

        if denominator <= self.epsilon:
            return 0.0

        normalized_concentration = (
            concentration - uniform_reference
        ) / denominator

        return self._bounded_float(
            normalized_concentration,
            "top_power_concentration",
        )

    def extract(
        self,
        frequency: ArrayLike,
        power: ArrayLike,
    ) -> dict[str, float]:
        """
        Extract spectral complexity features.

        Parameters
        frequency : array-like
            Lomb-Scargle frequency grid.
        power : array-like
            Lomb-Scargle periodogram power.

        Returns
        dict
            Flat dictionary with exactly 7 spectral complexity features.
        """
        distribution = self.build_power_distribution(frequency, power)

        return {
            "normalized_spectral_entropy": (
                self.normalized_spectral_entropy(distribution)
            ),
            "spectral_gini_index": self.spectral_gini_index(distribution),
            "normalized_power_variance": (
                self.normalized_power_variance(distribution)
            ),
            "power_skewness": self.power_skewness(distribution),
            "power_excess_kurtosis": (
                self.power_excess_kurtosis(distribution)
            ),
            "robust_power_dispersion": (
                self.robust_power_dispersion(distribution)
            ),
            "top_power_concentration": (
                self.top_power_concentration(distribution)
            ),
        }

    def _validate_distribution(
        self,
        distribution: PowerDistributionResult,
    ) -> None:
        if not isinstance(distribution, PowerDistributionResult):
            raise TypeError("distribution must be a PowerDistributionResult.")

        if distribution.point_count < 3:
            raise ValueError("distribution must contain at least 3 points.")

        if distribution.total_power <= self.epsilon:
            raise ValueError(
                "total spectral power must be greater than epsilon."
            )

        if not np.all(np.isfinite(distribution.nonnegative_power)):
            raise ValueError("nonnegative_power must contain finite values.")

        if not np.all(np.isfinite(distribution.normalized_power)):
            raise ValueError("normalized_power must contain finite values.")

    @staticmethod
    def _finite_float(value: object, name: str) -> float:
        scalar = float(value)

        if not np.isfinite(scalar):
            raise RuntimeError(f"{name} must be finite.")

        return scalar

    def _nonnegative_finite_float(self, value: object, name: str) -> float:
        scalar = self._finite_float(value, name)

        if scalar < 0:
            raise RuntimeError(f"{name} must be non-negative.")

        return scalar

    def _bounded_float(
        self,
        value: object,
        name: str,
        lower: float = 0.0,
        upper: float = 1.0,
    ) -> float:
        scalar = self._finite_float(value, name)
        numerical_tolerance = 1e-10

        if scalar < lower - numerical_tolerance:
            raise RuntimeError(f"{name} is below its theoretical lower bound.")

        if scalar > upper + numerical_tolerance:
            raise RuntimeError(f"{name} exceeds its theoretical upper bound.")

        return float(np.clip(scalar, lower, upper))
