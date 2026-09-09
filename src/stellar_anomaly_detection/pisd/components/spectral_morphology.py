"""Spectral morphology features for Physics-Informed Spectral Descriptor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import find_peaks, peak_widths

from stellar_anomaly_detection.constants import (
    PEAK_HEIGHT_FACTOR,
    PEAK_PROMINENCE_FACTOR,
    PEAK_WIDTH_RELATIVE_HEIGHT,
    SPECTRAL_EPSILON,
    USE_RAYLEIGH_PEAK_DISTANCE,
)


@dataclass(frozen=True)
class PeakAnalysisResult:
    """
    Peak analysis result for a Lomb-Scargle periodogram.

    Parameters
    peak_indices : np.ndarray
        Indices of detected significant peaks.
    peak_frequencies : np.ndarray
        Frequencies corresponding to detected peaks.
    peak_heights : np.ndarray
        Power values at detected peak locations.
    prominences : np.ndarray
        Peak prominences computed by SciPy.
    widths : np.ndarray
        Peak widths converted to frequency units.
    left_intersections : np.ndarray
        Interpolated left frequency intersections used for width estimation.
    right_intersections : np.ndarray
        Interpolated right frequency intersections used for width estimation.
    height_threshold : float
        Adaptive height threshold used for peak detection.
    prominence_threshold : float
        Adaptive prominence threshold used for peak detection.
    robust_background : float
        Robust spectral background estimate.
    robust_sigma : float
        Robust spectral scale estimate.
    minimum_peak_distance_samples : int
        Minimum distance between peaks, expressed in frequency bins.
    """

    peak_indices: NDArray[np.int64]
    peak_frequencies: NDArray[np.float64]
    peak_heights: NDArray[np.float64]
    prominences: NDArray[np.float64]
    widths: NDArray[np.float64]
    left_intersections: NDArray[np.float64]
    right_intersections: NDArray[np.float64]
    height_threshold: float
    prominence_threshold: float
    robust_background: float
    robust_sigma: float
    minimum_peak_distance_samples: int

    def to_dict(
        self,
    ) -> dict[
        str,
        NDArray[np.int64] | NDArray[np.float64] | float | int,
    ]:
        """
        Return the peak analysis result as a dictionary.

        Returns
        dict
            Dictionary representation. NumPy arrays are preserved.
        """
        return {
            "peak_indices": self.peak_indices,
            "peak_frequencies": self.peak_frequencies,
            "peak_heights": self.peak_heights,
            "prominences": self.prominences,
            "widths": self.widths,
            "left_intersections": self.left_intersections,
            "right_intersections": self.right_intersections,
            "height_threshold": self.height_threshold,
            "prominence_threshold": self.prominence_threshold,
            "robust_background": self.robust_background,
            "robust_sigma": self.robust_sigma,
            "minimum_peak_distance_samples": (
                self.minimum_peak_distance_samples
            ),
        }


class SpectralMorphology:
    """
    Extract spectral morphology features from a Lomb-Scargle periodogram.

    Notes
    Peak density, peak widths and peak separation features are directly
    comparable only across periodograms computed with consistent frequency
    units, bounds and resolution settings.
    """

    def __init__(
        self,
        peak_height_factor: float = PEAK_HEIGHT_FACTOR,
        peak_prominence_factor: float = PEAK_PROMINENCE_FACTOR,
        peak_width_relative_height: float = PEAK_WIDTH_RELATIVE_HEIGHT,
        epsilon: float = SPECTRAL_EPSILON,
        use_rayleigh_peak_distance: bool = USE_RAYLEIGH_PEAK_DISTANCE,
    ) -> None:
        self.peak_height_factor = float(peak_height_factor)
        self.peak_prominence_factor = float(peak_prominence_factor)
        self.peak_width_relative_height = float(peak_width_relative_height)
        self.epsilon = float(epsilon)
        self.use_rayleigh_peak_distance = use_rayleigh_peak_distance

        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate spectral morphology configuration."""
        if not np.isfinite(self.peak_height_factor):
            raise ValueError("peak_height_factor must be finite.")

        if self.peak_height_factor <= 0:
            raise ValueError("peak_height_factor must be greater than zero.")

        if not np.isfinite(self.peak_prominence_factor):
            raise ValueError("peak_prominence_factor must be finite.")

        if self.peak_prominence_factor <= 0:
            raise ValueError(
                "peak_prominence_factor must be greater than zero."
            )

        if not np.isfinite(self.peak_width_relative_height):
            raise ValueError("peak_width_relative_height must be finite.")

        if not 0 < self.peak_width_relative_height < 1:
            raise ValueError(
                "peak_width_relative_height must be strictly between 0 and 1."
            )

        if not np.isfinite(self.epsilon):
            raise ValueError("epsilon must be finite.")

        if self.epsilon <= 0:
            raise ValueError("epsilon must be greater than zero.")

        if not isinstance(self.use_rayleigh_peak_distance, bool):
            raise TypeError("use_rayleigh_peak_distance must be bool.")

    def validate_input(
        self,
        frequency: ArrayLike,
        power: ArrayLike,
        time_baseline: float | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float | None]:
        """
        Validate periodogram inputs.

        Parameters
        frequency : array-like
            Strictly positive and strictly increasing frequency grid.
        power : array-like
            Lomb-Scargle periodogram power values.
        time_baseline : float, optional
            Time baseline of the original light curve, ``max(time) - min(time)``.

        Returns
        tuple
            Validated copies of frequency, power and validated time baseline.

        Raises
        ValueError
            If inputs are malformed or contain invalid values.
        """
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

        validated_time_baseline = None
        if time_baseline is not None:
            validated_time_baseline = float(time_baseline)

            if not np.isfinite(validated_time_baseline):
                raise ValueError("time_baseline must be finite.")

            if validated_time_baseline <= 0:
                raise ValueError("time_baseline must be greater than zero.")

        return (
            frequency_array.astype(float, copy=True),
            power_array.astype(float, copy=True),
            validated_time_baseline,
        )

    def robust_background_and_scale(
        self,
        power: ArrayLike,
    ) -> tuple[float, float]:
        """
        Estimate robust spectral background and scale.

        Parameters
        power : array-like
            Periodogram power values.

        Returns
        tuple of float
            Robust background and robust scale.
        """
        power_array = self._validate_power(power)

        median_power = float(np.median(power_array))
        mad = float(np.median(np.abs(power_array - median_power)))
        sigma_mad = 1.4826 * mad

        if sigma_mad > self.epsilon:
            robust_sigma = sigma_mad
        else:
            iqr = float(
                np.percentile(power_array, 75)
                - np.percentile(power_array, 25)
            )
            sigma_iqr = iqr / 1.349
            robust_sigma = (
                sigma_iqr if sigma_iqr > self.epsilon else self.epsilon
            )

        return (
            self._finite_float(median_power, "robust_background"),
            self._finite_float(robust_sigma, "robust_sigma"),
        )

    def adaptive_thresholds(
        self,
        power: ArrayLike,
    ) -> tuple[float, float, float, float]:
        """Compute adaptive height and prominence thresholds.

        Parameters
        power : array-like
            Periodogram power values.

        Returns
        tuple of float
            Height threshold, prominence threshold, robust background and
            robust sigma.
        """
        robust_background, robust_sigma = self.robust_background_and_scale(
            power
        )
        height_threshold = (
            robust_background + self.peak_height_factor * robust_sigma
        )
        prominence_threshold = self.peak_prominence_factor * robust_sigma

        return (
            self._finite_float(height_threshold, "height_threshold"),
            self._finite_float(prominence_threshold, "prominence_threshold"),
            robust_background,
            robust_sigma,
        )

    def minimum_peak_distance(
        self,
        frequency: ArrayLike,
        time_baseline: float | None = None,
    ) -> int:
        """Compute the minimum distance between peaks in spectral bins.

        Parameters
        frequency : array-like
            Strictly increasing frequency grid.
        time_baseline : float, optional
            Time baseline of the original light curve.

        Returns
        int
            Minimum peak distance in bins.
        """
        frequency_array = self._validate_frequency(frequency)

        if not self.use_rayleigh_peak_distance or time_baseline is None:
            return 1

        validated_time_baseline = float(time_baseline)
        if not np.isfinite(validated_time_baseline):
            raise ValueError("time_baseline must be finite.")

        if validated_time_baseline <= 0:
            raise ValueError("time_baseline must be greater than zero.")

        delta_f_rayleigh = 1.0 / validated_time_baseline
        frequency_step = float(np.median(np.diff(frequency_array)))

        distance_samples = int(
            max(1, np.ceil(delta_f_rayleigh / frequency_step))
        )
        distance_samples = min(
            distance_samples,
            max(1, frequency_array.size - 1),
        )

        return distance_samples

    def detect_peaks(
        self,
        frequency: ArrayLike,
        power: ArrayLike,
        time_baseline: float | None = None,
    ) -> PeakAnalysisResult:
        """
        Detect significant peaks and estimate peak properties.

        Parameters
        frequency : array-like
            Lomb-Scargle frequency grid.
        power : array-like
            Lomb-Scargle periodogram power.
        time_baseline : float, optional
            Time baseline of the original light curve.

        Returns
        PeakAnalysisResult
            Structured peak analysis result.

        Notes
        Peak widths may be zero for degenerate or poorly resolved peaks.
        Such values are preserved and should be inspected during validation.
        """
        frequency_array, power_array, validated_time_baseline = (
            self.validate_input(frequency, power, time_baseline)
        )

        (
            height_threshold,
            prominence_threshold,
            robust_background,
            robust_sigma,
        ) = self.adaptive_thresholds(power_array)

        minimum_distance = self.minimum_peak_distance(
            frequency_array,
            validated_time_baseline,
        )

        peak_indices, properties = find_peaks(
            power_array,
            height=height_threshold,
            prominence=prominence_threshold,
            distance=minimum_distance,
        )
        peak_indices = peak_indices.astype(np.int64, copy=False)

        if peak_indices.size == 0:
            return self._empty_peak_result(
                height_threshold=height_threshold,
                prominence_threshold=prominence_threshold,
                robust_background=robust_background,
                robust_sigma=robust_sigma,
                minimum_peak_distance_samples=minimum_distance,
            )

        widths_bins, _, left_ips, right_ips = peak_widths(
            power_array,
            peak_indices,
            rel_height=self.peak_width_relative_height,
        )

        frequency_index = np.arange(frequency_array.size, dtype=float)
        left_frequencies = np.interp(left_ips, frequency_index, frequency_array)
        right_frequencies = np.interp(
            right_ips,
            frequency_index,
            frequency_array,
        )

        # Widths from peak_widths are fractional index widths; interpolation
        # preserves non-uniform frequency grids better than a median-step scale.
        widths_frequency = right_frequencies - left_frequencies

        peak_heights = np.asarray(
            properties["peak_heights"],
            dtype=float,
        )
        prominences = np.asarray(
            properties["prominences"],
            dtype=float,
        )

        return PeakAnalysisResult(
            peak_indices=peak_indices,
            peak_frequencies=frequency_array[peak_indices].astype(
                float,
                copy=True,
            ),
            peak_heights=peak_heights.astype(float, copy=True),
            prominences=prominences.astype(float, copy=True),
            widths=np.asarray(widths_frequency, dtype=float),
            left_intersections=np.asarray(left_frequencies, dtype=float),
            right_intersections=np.asarray(right_frequencies, dtype=float),
            height_threshold=height_threshold,
            prominence_threshold=prominence_threshold,
            robust_background=robust_background,
            robust_sigma=robust_sigma,
            minimum_peak_distance_samples=minimum_distance,
        )

    def peak_density(
        self,
        peak_count: int,
        frequency: ArrayLike,
    ) -> float:
        """Compute peak density over the frequency span."""
        frequency_array = self._validate_frequency(frequency)
        frequency_span = frequency_array[-1] - frequency_array[0]
        density = int(peak_count) / frequency_span
        return self._finite_float(density, "peak_density")

    def peak_width_statistics(
        self,
        widths: ArrayLike,
    ) -> tuple[float, float, float]:
        """Compute width mean, maximum and variability."""
        widths_array = np.asarray(widths, dtype=float)

        if widths_array.size == 0:
            return 0.0, 0.0, 0.0

        average_width = self._finite_float(
            np.mean(widths_array),
            "average_peak_width",
        )
        maximum_width = self._finite_float(
            np.max(widths_array),
            "maximum_peak_width",
        )
        width_variability = self.coefficient_of_variation(widths_array)

        return average_width, maximum_width, width_variability

    def prominence_statistics(
        self,
        prominences: ArrayLike,
    ) -> tuple[float, float, float]:
        """Compute prominence mean, maximum and variability."""
        prominence_array = np.asarray(prominences, dtype=float)

        if prominence_array.size == 0:
            return 0.0, 0.0, 0.0

        average_prominence = self._finite_float(
            np.mean(prominence_array),
            "average_prominence",
        )
        maximum_prominence = self._finite_float(
            np.max(prominence_array),
            "maximum_prominence",
        )
        prominence_variability = self.coefficient_of_variation(
            prominence_array
        )

        return (
            average_prominence,
            maximum_prominence,
            prominence_variability,
        )

    def peak_height_variability(
        self,
        peak_heights: ArrayLike,
    ) -> float:
        """Compute coefficient of variation of peak heights."""
        peak_heights_array = np.asarray(peak_heights, dtype=float)

        if peak_heights_array.size == 0:
            return 0.0

        return self.coefficient_of_variation(peak_heights_array)

    def peak_separation_statistics(
        self,
        peak_frequencies: ArrayLike,
    ) -> tuple[float, float]:
        """Compute mean separation and separation variability."""
        peak_frequency_array = np.asarray(peak_frequencies, dtype=float)

        if peak_frequency_array.size < 2:
            return 0.0, 0.0

        sorted_frequencies = np.sort(peak_frequency_array)
        separations = np.diff(sorted_frequencies)

        mean_separation = self._finite_float(
            np.mean(separations),
            "mean_peak_separation",
        )

        if separations.size == 1:
            return mean_separation, 0.0

        separation_variability = self.coefficient_of_variation(separations)
        return mean_separation, separation_variability

    def coefficient_of_variation(
        self,
        values: ArrayLike,
    ) -> float:
        """Compute population coefficient of variation."""
        values_array = np.asarray(values, dtype=float)

        if values_array.size <= 1:
            return 0.0

        mean_value = float(np.mean(values_array))
        denominator = max(abs(mean_value), self.epsilon)
        cv = np.std(values_array, ddof=0) / denominator

        return self._finite_float(cv, "coefficient_of_variation")

    def extract_from_analysis(
        self,
        frequency: ArrayLike,
        analysis: PeakAnalysisResult,
    ) -> dict[str, float | int]:
        """Extract morphology features from an existing peak analysis.

        Parameters
        ----------
        frequency : array-like
            Frequency grid used to produce ``analysis``.
        analysis : PeakAnalysisResult
            Previously computed peak analysis to reuse.

        Returns
        -------
        dict
            Flat dictionary with exactly 13 spectral morphology features.

        Raises
        ------
        TypeError
            If ``analysis`` is not a :class:`PeakAnalysisResult`.
        ValueError
            If the frequency grid or peak analysis is inconsistent.
        """
        frequency_array = self._validate_frequency(frequency)
        self._validate_peak_analysis(frequency_array, analysis)

        peak_count = int(analysis.peak_indices.size)
        has_peaks = int(peak_count >= 1)
        has_multiple_peaks = int(peak_count >= 2)

        if peak_count == 0:
            return {
                "has_peaks": 0,
                "has_multiple_peaks": 0,
                "peak_count": 0,
                "peak_density": 0.0,
                "average_peak_width": 0.0,
                "maximum_peak_width": 0.0,
                "peak_width_variability": 0.0,
                "average_prominence": 0.0,
                "maximum_prominence": 0.0,
                "prominence_variability": 0.0,
                "peak_height_variability": 0.0,
                "mean_peak_separation": 0.0,
                "peak_separation_variability": 0.0,
            }

        average_width, maximum_width, width_variability = (
            self.peak_width_statistics(analysis.widths)
        )
        average_prominence, maximum_prominence, prominence_variability = (
            self.prominence_statistics(analysis.prominences)
        )
        height_variability = self.peak_height_variability(
            analysis.peak_heights
        )
        mean_separation, separation_variability = (
            self.peak_separation_statistics(analysis.peak_frequencies)
        )

        return {
            "has_peaks": has_peaks,
            "has_multiple_peaks": has_multiple_peaks,
            "peak_count": peak_count,
            "peak_density": self.peak_density(
                peak_count,
                frequency_array,
            ),
            "average_peak_width": average_width,
            "maximum_peak_width": maximum_width,
            "peak_width_variability": width_variability,
            "average_prominence": average_prominence,
            "maximum_prominence": maximum_prominence,
            "prominence_variability": prominence_variability,
            "peak_height_variability": height_variability,
            "mean_peak_separation": mean_separation,
            "peak_separation_variability": separation_variability,
        }

    def extract(
        self,
        frequency: ArrayLike,
        power: ArrayLike,
        time_baseline: float | None = None,
    ) -> dict[str, float | int]:
        """Extract spectral morphology features.

        Parameters
        frequency : array-like
            Lomb-Scargle frequency grid.
        power : array-like
            Lomb-Scargle periodogram power.
        time_baseline : float, optional
            Time baseline of the original light curve.

        Returns
        dict
            Flat dictionary with exactly 13 spectral morphology features.
        """
        analysis = self.detect_peaks(
            frequency=frequency,
            power=power,
            time_baseline=time_baseline,
        )
        return self.extract_from_analysis(
            frequency=frequency,
            analysis=analysis,
        )

    @staticmethod
    def _validate_peak_analysis(
        frequency_array: NDArray[np.float64],
        analysis: PeakAnalysisResult,
    ) -> None:
        """Validate a peak analysis against its frequency grid."""
        if not isinstance(analysis, PeakAnalysisResult):
            raise TypeError("analysis must be a PeakAnalysisResult instance.")

        array_names = (
            "peak_indices",
            "peak_frequencies",
            "peak_heights",
            "prominences",
            "widths",
            "left_intersections",
            "right_intersections",
        )
        arrays = {
            name: np.asarray(getattr(analysis, name)) for name in array_names
        }
        for name, array in arrays.items():
            if array.ndim != 1:
                raise ValueError(f"analysis.{name} must be a 1D array.")

        lengths = {array.size for array in arrays.values()}
        if len(lengths) != 1:
            raise ValueError(
                "All PeakAnalysisResult arrays must have the same length."
            )

        peak_indices = arrays["peak_indices"]
        if not np.issubdtype(peak_indices.dtype, np.integer):
            raise ValueError("analysis.peak_indices must contain integers.")
        if np.any(peak_indices < 0):
            raise ValueError("analysis.peak_indices must be non-negative.")
        if np.any(peak_indices >= frequency_array.size):
            raise ValueError(
                "analysis.peak_indices must be within the frequency grid."
            )

        float_array_names = array_names[1:]
        for name in float_array_names:
            if not np.all(np.isfinite(arrays[name])):
                raise ValueError(
                    f"analysis.{name} must contain only finite values."
                )

        peak_frequencies = arrays["peak_frequencies"]
        if np.any(peak_frequencies <= 0):
            raise ValueError(
                "analysis.peak_frequencies must be strictly positive."
            )
        if not np.allclose(
            peak_frequencies,
            frequency_array[peak_indices],
        ):
            raise ValueError(
                "analysis.peak_frequencies do not match the frequency grid."
            )

        scalar_names = (
            "height_threshold",
            "prominence_threshold",
            "robust_background",
            "robust_sigma",
        )
        for name in scalar_names:
            if not np.isfinite(float(getattr(analysis, name))):
                raise ValueError(f"analysis.{name} must be finite.")

    @staticmethod
    def _empty_peak_result(
        height_threshold: float,
        prominence_threshold: float,
        robust_background: float,
        robust_sigma: float,
        minimum_peak_distance_samples: int,
    ) -> PeakAnalysisResult:
        return PeakAnalysisResult(
            peak_indices=np.array([], dtype=np.int64),
            peak_frequencies=np.array([], dtype=np.float64),
            peak_heights=np.array([], dtype=np.float64),
            prominences=np.array([], dtype=np.float64),
            widths=np.array([], dtype=np.float64),
            left_intersections=np.array([], dtype=np.float64),
            right_intersections=np.array([], dtype=np.float64),
            height_threshold=float(height_threshold),
            prominence_threshold=float(prominence_threshold),
            robust_background=float(robust_background),
            robust_sigma=float(robust_sigma),
            minimum_peak_distance_samples=int(minimum_peak_distance_samples),
        )

    @staticmethod
    def _validate_frequency(
        frequency: ArrayLike,
    ) -> NDArray[np.float64]:
        frequency_array = np.asarray(frequency, dtype=float)

        if frequency_array.ndim != 1:
            raise ValueError("frequency must be a 1D array.")

        if frequency_array.size < 3:
            raise ValueError("At least three frequency points are required.")

        if not np.all(np.isfinite(frequency_array)):
            raise ValueError("frequency must contain only finite values.")

        if np.any(frequency_array <= 0):
            raise ValueError("frequency values must be strictly positive.")

        if np.any(np.diff(frequency_array) <= 0):
            raise ValueError("frequency values must be strictly increasing.")

        return frequency_array.astype(float, copy=True)

    @staticmethod
    def _validate_power(
        power: ArrayLike,
    ) -> NDArray[np.float64]:
        power_array = np.asarray(power, dtype=float)

        if power_array.ndim != 1:
            raise ValueError("power must be a 1D array.")

        if power_array.size == 0:
            raise ValueError("power must not be empty.")

        if not np.all(np.isfinite(power_array)):
            raise ValueError("power must contain only finite values.")

        return power_array.astype(float, copy=True)

    @staticmethod
    def _finite_float(value: object, name: str) -> float:
        scalar = float(value)

        if not np.isfinite(scalar):
            raise RuntimeError(f"{name} must be finite.")

        return scalar
