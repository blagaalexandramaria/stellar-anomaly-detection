import numpy as np

from stellar_anomaly_detection.pisd.components.spectral_stability import SpectralStability
from stellar_anomaly_detection.pisd.policy import (
    STABILITY_N_PERTURBATIONS,
    derive_observation_stability_seed,
)
from stellar_anomaly_detection.spectral import LombScargleAnalyzer


class RecordingAnalyzer(LombScargleAnalyzer):
    def __init__(self):
        super().__init__(maximum_frequency=1.0, false_alarm_probability=None)
        self.grids = []

    def compute_power_on_grid(self, time, flux, frequency, flux_error=None):
        self.grids.append(np.asarray(frequency).copy())
        return np.asarray([1.0, 2.0, 1.0])


def test_stability_seed_is_deterministic_and_object_specific():
    first = derive_observation_stability_seed("object:a", "observation:a")
    assert first == derive_observation_stability_seed("object:a", "observation:a")
    assert first != derive_observation_stability_seed("object:b", "observation:a")
    assert first != derive_observation_stability_seed("object:a", "observation:b")
    assert STABILITY_N_PERTURBATIONS == 100


def test_stability_configuration_controls_count_and_reuses_fixed_grid():
    analyzer = RecordingAnalyzer()
    stability = SpectralStability(analyzer, n_perturbations=3, random_state=7)
    time = np.asarray([0.0, 1.0, 2.0]); flux = np.asarray([0.0, 1.0, 0.0])
    frequency = np.asarray([0.1, 0.2, 0.3]); power = np.asarray([1.0, 2.0, 1.0])
    noise = stability.estimate_noise_scale(flux, None)
    first = stability.run_perturbations(time, flux, None, frequency, power, .2, 2.0,
                                        noise, 2.0, np.random.default_rng(7))
    second = stability.run_perturbations(time, flux, None, frequency, power, .2, 2.0,
                                         noise, 2.0, np.random.default_rng(7))
    assert all(value.shape == (3,) for value in first)
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    assert len(analyzer.grids) == 6
    assert all(np.array_equal(grid, frequency) for grid in analyzer.grids)
