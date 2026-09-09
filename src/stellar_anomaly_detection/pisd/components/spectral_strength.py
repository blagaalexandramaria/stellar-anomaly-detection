"""Spectral strength features for Physics-Informed Spectral Descriptor."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class SpectralStrength:
    """
    Extract spectral strength features from a Lomb-Scargle periodogram.

    Notes
    The extracted features are intended for Lomb-Scargle periodograms
    computed using a fixed normalization across the entire dataset.
    The PISD pipeline currently assumes the ``standard`` normalization.
    """

    def validate_input(
        self,
        frequency: ArrayLike,
        power: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Validate periodogram frequency and power arrays.

        Parameters
        frequency : array-like
            Strictly positive, strictly increasing frequency values.
        power : array-like
            Periodogram power values.

        Returns
        tuple of np.ndarray
            Validated copies of ``frequency`` and ``power``.

        Raises
        ValueError
            If arrays are invalid, non-finite, empty, too short, mismatched,
            or if frequencies are not strictly positive and increasing.
        """
        frequency_array = np.asarray(frequency, dtype=float)
        power_array = np.asarray(power, dtype=float)

        if frequency_array.ndim != 1 or power_array.ndim != 1:
            raise ValueError("frequency and power must be 1D arrays.")

        if frequency_array.size == 0 or power_array.size == 0:
            raise ValueError("frequency and power must not be empty.")

        if frequency_array.size != power_array.size:
            raise ValueError("frequency and power must have the same length.")

        if frequency_array.size < 2:
            raise ValueError("At least two spectral points are required.")

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

    def maximum_power(self, power: ArrayLike) -> float:
        """
        Return the maximum spectral power.

        Parameters
        power : array-like
            Periodogram power values.

        Returns
        float
            Maximum power.
        """
        power_array = self._validate_power(power)
        return self._finite_float(np.max(power_array), "maximum_power")

    def mean_power(self, power: ArrayLike) -> float:
        """
        Return the mean spectral power.

        Parameters
        power : array-like
            Periodogram power values.

        Returns
        float
            Mean power.
        """
        power_array = self._validate_power(power)
        return self._finite_float(np.mean(power_array), "mean_power")

    def integrated_spectral_power(
        self,
        frequency: ArrayLike,
        power: ArrayLike,
    ) -> float:
        """
        Return the integrated spectral power.

        Parameters
        frequency : array-like
            Frequency grid.
        power : array-like
            Periodogram power values.

        Returns
        float
            Spectral power integrated over frequency.
        """
        frequency_array, power_array = self.validate_input(frequency, power)
        return self._integrated_spectral_power_validated(
            frequency_array,
            power_array,
        )

    def robust_spectral_snr(
        self,
        power: ArrayLike,
        epsilon: float = 1e-12,
    ) -> float:
        """
        Return a robust spectral signal-to-noise ratio.

        Parameters
        power : array-like
            Periodogram power values.
        epsilon : float, default=1e-12
            Positive lower bound used to avoid division by zero.

        Returns
        float
            Robust spectral SNR.
        """
        power_array = self._validate_power(power)
        return self._robust_spectral_snr_validated(power_array, epsilon)

    def peak_bin_concentration(
        self,
        power: ArrayLike,
        epsilon: float = 1e-12,
    ) -> float:
        """
        Return the concentration of power in the strongest bin.

        Parameters
        power : array-like
            Periodogram power values.
        epsilon : float, default=1e-12
            Positive lower bound used when total power is near zero.

        Returns
        float
            Peak-bin concentration.

        Notes
        This feature assumes non-negative periodogram power values and should
        only be compared across periodograms computed with the same frequency
        grid configuration and normalization.
        """
        power_array = self._validate_power(power)
        return self._peak_bin_concentration_validated(power_array, epsilon)

    def extract(
        self,
        frequency: ArrayLike,
        power: ArrayLike,
    ) -> dict[str, float]:
        """
        Extract all spectral strength features.

        Parameters
        frequency : array-like
            Frequency grid produced by Lomb-Scargle analysis.
        power : array-like
            Power values produced by Lomb-Scargle analysis.

        Returns
        dict
            Dictionary with spectral strength features.
        """
        frequency_array, power_array = self.validate_input(frequency, power)

        return {
            "maximum_power": self._maximum_power_validated(power_array),
            "mean_power": self._mean_power_validated(power_array),
            "integrated_spectral_power": (
                self._integrated_spectral_power_validated(
                    frequency_array,
                    power_array,
                )
            ),
            "robust_spectral_snr": self._robust_spectral_snr_validated(
                power_array,
            ),
            "peak_bin_concentration": self._peak_bin_concentration_validated(
                power_array,
            ),
        }

    @staticmethod
    def _validate_power(power: ArrayLike) -> NDArray[np.float64]:
        power_array = np.asarray(power, dtype=float)

        if power_array.ndim != 1:
            raise ValueError("power must be a 1D array.")

        if power_array.size == 0:
            raise ValueError("power must not be empty.")

        if not np.all(np.isfinite(power_array)):
            raise ValueError("power must contain only finite values.")

        return power_array.astype(float, copy=True)

    @staticmethod
    def _validate_epsilon(epsilon: float) -> float:
        epsilon_value = float(epsilon)

        if not np.isfinite(epsilon_value):
            raise ValueError("epsilon must be finite.")

        if epsilon_value <= 0:
            raise ValueError("epsilon must be greater than zero.")

        return epsilon_value

    def _maximum_power_validated(
        self,
        power: NDArray[np.float64],
    ) -> float:
        return self._finite_float(np.max(power), "maximum_power")

    def _mean_power_validated(
        self,
        power: NDArray[np.float64],
    ) -> float:
        return self._finite_float(np.mean(power), "mean_power")

    def _integrated_spectral_power_validated(
        self,
        frequency: NDArray[np.float64],
        power: NDArray[np.float64],
    ) -> float:
        # Integrated spectral power depends on frequency spacing; unlike
        # sum(power), it preserves each spectral interval's width.
        value = np.trapezoid(power, frequency)
        return self._finite_float(value, "integrated_spectral_power")

    def _robust_spectral_snr_validated(
        self,
        power: NDArray[np.float64],
        epsilon: float = 1e-12,
    ) -> float:
        epsilon_value = self._validate_epsilon(epsilon)

        median_power = np.median(power)
        mad = np.median(np.abs(power - median_power))
        robust_sigma = 1.4826 * mad
        denominator = max(float(robust_sigma), epsilon_value)
        snr = (np.max(power) - median_power) / denominator

        return self._finite_float(snr, "robust_spectral_snr")

    def _peak_bin_concentration_validated(
        self,
        power: NDArray[np.float64],
        epsilon: float = 1e-12,
    ) -> float:
        epsilon_value = self._validate_epsilon(epsilon)

        denominator = max(float(np.sum(power)), epsilon_value)
        concentration = np.max(power) / denominator

        return self._finite_float(
            concentration,
            "peak_bin_concentration",
        )

    @staticmethod
    def _finite_float(value: object, name: str) -> float:
        scalar = float(value)

        if not np.isfinite(scalar):
            raise RuntimeError(f"{name} must be finite.")

        return scalar
