"""Harmonic structure features for Physics-Informed Spectral Descriptor."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from stellar_anomaly_detection.pisd.policy import (
    HARMONIC_EPSILON,
    HARMONIC_GRID_TOLERANCE_FACTOR,
    HARMONIC_TOLERANCE_FACTOR,
    MAX_FUNDAMENTAL_CANDIDATES,
    MAX_HARMONIC_ORDER,
)
from stellar_anomaly_detection.pisd.components.spectral_morphology import PeakAnalysisResult


@dataclass(frozen=True)
class HarmonicAnalysisResult:
    """Structured result of harmonic analysis."""

    fundamental_peak_position: int
    fundamental_peak_index: int
    fundamental_frequency: float
    fundamental_power: float
    harmonic_orders: NDArray[np.int64]
    harmonic_peak_positions: NDArray[np.int64]
    harmonic_peak_indices: NDArray[np.int64]
    harmonic_frequencies: NDArray[np.float64]
    harmonic_powers: NDArray[np.float64]
    expected_harmonic_frequencies: NDArray[np.float64]
    absolute_frequency_deviations: NDArray[np.float64]
    normalized_frequency_deviations: NDArray[np.float64]
    tolerance: float
    candidate_count: int
    harmonic_count: int

    def to_dict(
        self,
    ) -> dict[str, int | float | NDArray[np.int64] | NDArray[np.float64]]:
        """Return the harmonic analysis result as a dictionary.

        Returns
        -------
        dict
            Dictionary representation. NumPy arrays are preserved.
        """
        return {
            "fundamental_peak_position": self.fundamental_peak_position,
            "fundamental_peak_index": self.fundamental_peak_index,
            "fundamental_frequency": self.fundamental_frequency,
            "fundamental_power": self.fundamental_power,
            "harmonic_orders": self.harmonic_orders,
            "harmonic_peak_positions": self.harmonic_peak_positions,
            "harmonic_peak_indices": self.harmonic_peak_indices,
            "harmonic_frequencies": self.harmonic_frequencies,
            "harmonic_powers": self.harmonic_powers,
            "expected_harmonic_frequencies": (
                self.expected_harmonic_frequencies
            ),
            "absolute_frequency_deviations": (
                self.absolute_frequency_deviations
            ),
            "normalized_frequency_deviations": (
                self.normalized_frequency_deviations
            ),
            "tolerance": self.tolerance,
            "candidate_count": self.candidate_count,
            "harmonic_count": self.harmonic_count,
        }


class HarmonicStructure:
    """Analyze harmonic organization among already detected spectral peaks.

    Notes
    -----
    For scientifically meaningful harmonic matching, ``time_baseline`` should
    normally be provided. When it is unavailable, the fallback tolerance is
    estimated from detected peak spacing and should be treated as an
    approximate computational fallback rather than a physical frequency
    resolution.
    """

    def __init__(
        self,
        max_harmonic_order: int = MAX_HARMONIC_ORDER,
        max_fundamental_candidates: int = MAX_FUNDAMENTAL_CANDIDATES,
        harmonic_tolerance_factor: float = HARMONIC_TOLERANCE_FACTOR,
        harmonic_grid_tolerance_factor: float = HARMONIC_GRID_TOLERANCE_FACTOR,
        epsilon: float = HARMONIC_EPSILON,
    ) -> None:
        self.max_harmonic_order = max_harmonic_order
        self.max_fundamental_candidates = max_fundamental_candidates
        self.harmonic_tolerance_factor = float(harmonic_tolerance_factor)
        self.harmonic_grid_tolerance_factor = float(
            harmonic_grid_tolerance_factor
        )
        self.epsilon = float(epsilon)

        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate harmonic structure configuration."""
        if isinstance(self.max_harmonic_order, bool) or not isinstance(
            self.max_harmonic_order,
            (int, np.integer),
        ):
            raise TypeError("max_harmonic_order must be an integer.")

        if self.max_harmonic_order < 2:
            raise ValueError("max_harmonic_order must be at least 2.")

        self.max_harmonic_order = int(self.max_harmonic_order)

        if isinstance(self.max_fundamental_candidates, bool) or not isinstance(
            self.max_fundamental_candidates,
            (int, np.integer),
        ):
            raise TypeError("max_fundamental_candidates must be an integer.")

        if self.max_fundamental_candidates < 1:
            raise ValueError("max_fundamental_candidates must be at least 1.")

        self.max_fundamental_candidates = int(
            self.max_fundamental_candidates
        )

        if not np.isfinite(self.harmonic_tolerance_factor):
            raise ValueError("harmonic_tolerance_factor must be finite.")

        if self.harmonic_tolerance_factor <= 0:
            raise ValueError(
                "harmonic_tolerance_factor must be greater than zero."
            )

        if not np.isfinite(self.harmonic_grid_tolerance_factor):
            raise ValueError("harmonic_grid_tolerance_factor must be finite.")

        if self.harmonic_grid_tolerance_factor <= 0:
            raise ValueError(
                "harmonic_grid_tolerance_factor must be greater than zero."
            )

        if not np.isfinite(self.epsilon):
            raise ValueError("epsilon must be finite.")

        if self.epsilon <= 0:
            raise ValueError("epsilon must be greater than zero.")

    def validate_input(
        self,
        peak_analysis: PeakAnalysisResult,
        time_baseline: float | None = None,
        frequency_maximum: float | None = None,
    ) -> tuple[PeakAnalysisResult, float | None, float | None]:
        """Validate harmonic analysis inputs."""
        if not isinstance(peak_analysis, PeakAnalysisResult):
            raise TypeError("peak_analysis must be a PeakAnalysisResult.")

        arrays = (
            peak_analysis.peak_indices,
            peak_analysis.peak_frequencies,
            peak_analysis.peak_heights,
            peak_analysis.prominences,
            peak_analysis.widths,
            peak_analysis.left_intersections,
            peak_analysis.right_intersections,
        )

        if any(array.ndim != 1 for array in arrays):
            raise ValueError("All peak analysis arrays must be 1D.")

        peak_count = peak_analysis.peak_indices.size
        if not (
            peak_analysis.peak_frequencies.size
            == peak_analysis.peak_heights.size
            == peak_analysis.prominences.size
            == peak_count
        ):
            raise ValueError(
                "peak_indices, peak_frequencies, peak_heights and "
                "prominences must have the same length."
            )

        if not (
            peak_analysis.widths.size
            == peak_analysis.left_intersections.size
            == peak_analysis.right_intersections.size
            == peak_count
        ):
            raise ValueError(
                "Peak width arrays must have the same length as peak_indices."
            )

        if peak_count > 0:
            if np.any(peak_analysis.peak_indices < 0):
                raise ValueError("peak_indices must be non-negative.")

            if not np.all(np.isfinite(peak_analysis.peak_frequencies)):
                raise ValueError("peak_frequencies must be finite.")

            if np.any(peak_analysis.peak_frequencies <= 0):
                raise ValueError(
                    "peak_frequencies must be strictly positive."
                )

            if not np.all(np.isfinite(peak_analysis.peak_heights)):
                raise ValueError("peak_heights must be finite.")

            if not np.all(np.isfinite(peak_analysis.prominences)):
                raise ValueError("prominences must be finite.")

            if not np.all(np.isfinite(peak_analysis.widths)):
                raise ValueError("widths must be finite.")

            if not np.all(np.isfinite(peak_analysis.left_intersections)):
                raise ValueError("left_intersections must be finite.")

            if not np.all(np.isfinite(peak_analysis.right_intersections)):
                raise ValueError("right_intersections must be finite.")

        validated_time_baseline = None
        if time_baseline is not None:
            validated_time_baseline = float(time_baseline)

            if not np.isfinite(validated_time_baseline):
                raise ValueError("time_baseline must be finite.")

            if validated_time_baseline <= 0:
                raise ValueError("time_baseline must be greater than zero.")

        validated_frequency_maximum = None
        if frequency_maximum is not None:
            validated_frequency_maximum = float(frequency_maximum)

            if not np.isfinite(validated_frequency_maximum):
                raise ValueError("frequency_maximum must be finite.")

            if validated_frequency_maximum <= 0:
                raise ValueError("frequency_maximum must be greater than zero.")

            if peak_count > 0 and validated_frequency_maximum < float(
                np.max(peak_analysis.peak_frequencies)
            ):
                raise ValueError(
                    "frequency_maximum must be at least the maximum "
                    "detected peak frequency."
                )

        return (
            peak_analysis,
            validated_time_baseline,
            validated_frequency_maximum,
        )

    def harmonic_tolerance(
        self,
        peak_frequencies: NDArray[np.float64],
        time_baseline: float | None = None,
    ) -> float:
        """Compute harmonic matching tolerance."""
        frequencies = np.asarray(peak_frequencies, dtype=float)

        if frequencies.ndim != 1:
            raise ValueError("peak_frequencies must be a 1D array.")

        if frequencies.size > 0:
            if not np.all(np.isfinite(frequencies)):
                raise ValueError("peak_frequencies must be finite.")

            if np.any(frequencies <= 0):
                raise ValueError(
                    "peak_frequencies must be strictly positive."
                )

        if time_baseline is not None:
            baseline = float(time_baseline)

            if not np.isfinite(baseline):
                raise ValueError("time_baseline must be finite.")

            if baseline <= 0:
                raise ValueError("time_baseline must be greater than zero.")

            tolerance = self.harmonic_tolerance_factor / baseline
            return self._positive_finite_float(tolerance, "tolerance")

        unique_frequencies = np.unique(frequencies)
        if unique_frequencies.size < 2:
            return float(self.epsilon)

        frequency_step = float(np.median(np.diff(np.sort(unique_frequencies))))
        tolerance = self.harmonic_grid_tolerance_factor * frequency_step

        return self._positive_finite_float(tolerance, "tolerance")

    def _match_harmonics_for_candidate(
        self,
        fundamental_position: int,
        peak_frequencies: NDArray[np.float64],
        peak_heights: NDArray[np.float64],
        peak_indices: NDArray[np.int64],
        tolerance: float,
        frequency_maximum: float | None,
    ) -> HarmonicAnalysisResult:
        f0 = float(peak_frequencies[fundamental_position])
        p0 = float(peak_heights[fundamental_position])
        fundamental_index = int(peak_indices[fundamental_position])

        used_positions = {int(fundamental_position)}
        harmonic_orders: list[int] = []
        harmonic_positions: list[int] = []
        harmonic_indices: list[int] = []
        harmonic_frequencies: list[float] = []
        harmonic_powers: list[float] = []
        expected_frequencies: list[float] = []
        absolute_deviations: list[float] = []
        normalized_deviations: list[float] = []

        all_positions = np.arange(peak_frequencies.size)

        for order in range(2, self.max_harmonic_order + 1):
            expected_frequency = order * f0

            if (
                frequency_maximum is not None
                and expected_frequency > frequency_maximum
            ):
                break

            available_positions = np.array(
                [
                    position
                    for position in all_positions
                    if (
                        int(position) not in used_positions
                        and peak_frequencies[position] > f0
                        and abs(
                            peak_frequencies[position] - expected_frequency
                        )
                        <= tolerance
                    )
                ],
                dtype=np.int64,
            )

            if available_positions.size == 0:
                continue

            deviations = np.abs(
                peak_frequencies[available_positions] - expected_frequency
            )
            best_local_position = int(np.argmin(deviations))
            best_position = int(available_positions[best_local_position])
            absolute_deviation = float(deviations[best_local_position])

            used_positions.add(best_position)
            harmonic_orders.append(order)
            harmonic_positions.append(best_position)
            harmonic_indices.append(int(peak_indices[best_position]))
            harmonic_frequencies.append(float(peak_frequencies[best_position]))
            harmonic_powers.append(float(peak_heights[best_position]))
            expected_frequencies.append(float(expected_frequency))
            absolute_deviations.append(absolute_deviation)
            normalized_deviations.append(absolute_deviation / tolerance)

        harmonic_count = len(harmonic_orders)

        return HarmonicAnalysisResult(
            fundamental_peak_position=int(fundamental_position),
            fundamental_peak_index=fundamental_index,
            fundamental_frequency=self._finite_float(
                f0,
                "fundamental_frequency",
            ),
            fundamental_power=self._finite_float(p0, "fundamental_power"),
            harmonic_orders=np.asarray(harmonic_orders, dtype=np.int64),
            harmonic_peak_positions=np.asarray(
                harmonic_positions,
                dtype=np.int64,
            ),
            harmonic_peak_indices=np.asarray(harmonic_indices, dtype=np.int64),
            harmonic_frequencies=np.asarray(
                harmonic_frequencies,
                dtype=np.float64,
            ),
            harmonic_powers=np.asarray(harmonic_powers, dtype=np.float64),
            expected_harmonic_frequencies=np.asarray(
                expected_frequencies,
                dtype=np.float64,
            ),
            absolute_frequency_deviations=np.asarray(
                absolute_deviations,
                dtype=np.float64,
            ),
            normalized_frequency_deviations=np.asarray(
                normalized_deviations,
                dtype=np.float64,
            ),
            tolerance=self._positive_finite_float(tolerance, "tolerance"),
            candidate_count=0,
            harmonic_count=harmonic_count,
        )

    def select_fundamental(
        self,
        peak_analysis: PeakAnalysisResult,
        time_baseline: float | None = None,
        frequency_maximum: float | None = None,
    ) -> HarmonicAnalysisResult:
        """Select the fundamental peak and match harmonic peaks."""
        peak_analysis, validated_time_baseline, validated_frequency_maximum = (
            self.validate_input(
                peak_analysis,
                time_baseline,
                frequency_maximum,
            )
        )

        tolerance = self.harmonic_tolerance(
            peak_analysis.peak_frequencies,
            validated_time_baseline,
        )

        return self._select_fundamental_validated(
            peak_analysis=peak_analysis,
            tolerance=tolerance,
            frequency_maximum=validated_frequency_maximum,
        )

    def analyze(
        self,
        peak_analysis: PeakAnalysisResult,
        time_baseline: float | None = None,
        frequency_maximum: float | None = None,
    ) -> HarmonicAnalysisResult:
        """Run full harmonic structure analysis."""
        peak_analysis, validated_time_baseline, validated_frequency_maximum = (
            self.validate_input(
                peak_analysis,
                time_baseline,
                frequency_maximum,
            )
        )

        tolerance = self.harmonic_tolerance(
            peak_analysis.peak_frequencies,
            validated_time_baseline,
        )

        return self._select_fundamental_validated(
            peak_analysis=peak_analysis,
            tolerance=tolerance,
            frequency_maximum=validated_frequency_maximum,
        )

    def fundamental_dominance(
        self,
        analysis: HarmonicAnalysisResult,
    ) -> float:
        """Compute the fraction of harmonic-system power in the fundamental."""
        if analysis.fundamental_peak_position < 0:
            return 0.0

        harmonic_power = float(np.sum(analysis.harmonic_powers))
        total_power = analysis.fundamental_power + harmonic_power

        dominance = analysis.fundamental_power / max(total_power, self.epsilon)
        return self._finite_float(dominance, "fundamental_dominance")

    def first_harmonic_ratio(
        self,
        analysis: HarmonicAnalysisResult,
    ) -> float:
        """Compute the power ratio of the order-2 harmonic to fundamental."""
        if analysis.harmonic_count == 0:
            return 0.0

        matches = np.where(analysis.harmonic_orders == 2)[0]
        if matches.size == 0:
            return 0.0

        ratio = analysis.harmonic_powers[int(matches[0])] / max(
            analysis.fundamental_power,
            self.epsilon,
        )
        return self._finite_float(ratio, "first_harmonic_ratio")

    def strongest_harmonic_ratio(
        self,
        analysis: HarmonicAnalysisResult,
    ) -> float:
        """Compute the strongest harmonic power relative to fundamental."""
        if analysis.harmonic_count == 0:
            return 0.0

        ratio = np.max(analysis.harmonic_powers) / max(
            analysis.fundamental_power,
            self.epsilon,
        )
        return self._finite_float(ratio, "strongest_harmonic_ratio")

    def harmonic_complexity(
        self,
        analysis: HarmonicAnalysisResult,
    ) -> float:
        """Compute total harmonic power relative to fundamental."""
        if analysis.harmonic_count == 0:
            return 0.0

        complexity = np.sum(analysis.harmonic_powers) / max(
            analysis.fundamental_power,
            self.epsilon,
        )
        return self._finite_float(complexity, "harmonic_complexity")

    def harmonic_power_fraction(
        self,
        analysis: HarmonicAnalysisResult,
    ) -> float:
        """Compute harmonic power fraction within the harmonic system."""
        if analysis.fundamental_peak_position < 0:
            return 0.0

        harmonic_power = float(np.sum(analysis.harmonic_powers))
        total_power = analysis.fundamental_power + harmonic_power
        fraction = harmonic_power / max(total_power, self.epsilon)

        return self._finite_float(fraction, "harmonic_power_fraction")

    def mean_harmonic_deviation(
        self,
        analysis: HarmonicAnalysisResult,
    ) -> float:
        """Compute mean normalized harmonic frequency deviation."""
        if analysis.harmonic_count == 0:
            return 0.0

        deviation = np.mean(analysis.normalized_frequency_deviations)
        return self._finite_float(deviation, "mean_harmonic_deviation")

    def extract(
        self,
        peak_analysis: PeakAnalysisResult,
        time_baseline: float | None = None,
        frequency_maximum: float | None = None,
    ) -> dict[str, float | int]:
        """Extract harmonic structure features.

        Returns
        -------
        dict
            Flat dictionary with exactly 10 harmonic structure features.
        """
        analysis = self.analyze(
            peak_analysis=peak_analysis,
            time_baseline=time_baseline,
            frequency_maximum=frequency_maximum,
        )

        return {
            "has_harmonic_structure": int(analysis.harmonic_count >= 1),
            "harmonic_count": int(analysis.harmonic_count),
            "fundamental_frequency": self._finite_float(
                analysis.fundamental_frequency,
                "fundamental_frequency",
            ),
            "fundamental_power": self._finite_float(
                analysis.fundamental_power,
                "fundamental_power",
            ),
            "fundamental_dominance": self.fundamental_dominance(analysis),
            "first_harmonic_ratio": self.first_harmonic_ratio(analysis),
            "strongest_harmonic_ratio": self.strongest_harmonic_ratio(
                analysis
            ),
            "harmonic_complexity": self.harmonic_complexity(analysis),
            "harmonic_power_fraction": self.harmonic_power_fraction(analysis),
            "mean_harmonic_deviation": self.mean_harmonic_deviation(analysis),
        }

    def _select_fundamental_validated(
        self,
        peak_analysis: PeakAnalysisResult,
        tolerance: float,
        frequency_maximum: float | None,
    ) -> HarmonicAnalysisResult:
        peak_count = int(peak_analysis.peak_indices.size)

        if peak_count == 0:
            return self._empty_result(tolerance=tolerance, candidate_count=0)

        sorted_positions = np.argsort(peak_analysis.peak_frequencies)
        candidate_count = min(self.max_fundamental_candidates, peak_count)
        candidate_positions = sorted_positions[:candidate_count]

        best_result: HarmonicAnalysisResult | None = None
        best_key: tuple[int, float, float, float] | None = None

        for position in candidate_positions:
            result = self._match_harmonics_for_candidate(
                fundamental_position=int(position),
                peak_frequencies=peak_analysis.peak_frequencies,
                peak_heights=peak_analysis.peak_heights,
                peak_indices=peak_analysis.peak_indices,
                tolerance=tolerance,
                frequency_maximum=frequency_maximum,
            )
            result = replace(result, candidate_count=candidate_count)

            if result.harmonic_count > 0:
                mean_deviation = float(
                    np.mean(result.normalized_frequency_deviations)
                )
            else:
                mean_deviation = 0.0

            key = (
                result.harmonic_count,
                result.fundamental_power,
                -mean_deviation,
                -result.fundamental_frequency,
            )

            if best_key is None or key > best_key:
                best_key = key
                best_result = result

        if best_result is None:
            return self._empty_result(
                tolerance=tolerance,
                candidate_count=0,
            )

        return best_result

    def _empty_result(
        self,
        tolerance: float,
        candidate_count: int,
    ) -> HarmonicAnalysisResult:
        return HarmonicAnalysisResult(
            fundamental_peak_position=-1,
            fundamental_peak_index=-1,
            fundamental_frequency=0.0,
            fundamental_power=0.0,
            harmonic_orders=np.array([], dtype=np.int64),
            harmonic_peak_positions=np.array([], dtype=np.int64),
            harmonic_peak_indices=np.array([], dtype=np.int64),
            harmonic_frequencies=np.array([], dtype=np.float64),
            harmonic_powers=np.array([], dtype=np.float64),
            expected_harmonic_frequencies=np.array([], dtype=np.float64),
            absolute_frequency_deviations=np.array([], dtype=np.float64),
            normalized_frequency_deviations=np.array([], dtype=np.float64),
            tolerance=self._positive_finite_float(tolerance, "tolerance"),
            candidate_count=int(candidate_count),
            harmonic_count=0,
        )

    @staticmethod
    def _finite_float(value: object, name: str) -> float:
        scalar = float(value)

        if not np.isfinite(scalar):
            raise RuntimeError(f"{name} must be finite.")

        return scalar

    def _positive_finite_float(self, value: object, name: str) -> float:
        scalar = self._finite_float(value, name)

        if scalar <= 0:
            raise RuntimeError(f"{name} must be greater than zero.")

        return scalar
