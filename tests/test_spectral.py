import numpy as np

from stellar_anomaly_detection.spectral import LombScargleAnalyzer, analyze_spectral_window


def test_lomb_scargle_and_window_small_fixture():
    time = np.linspace(0.0, 20.0, 200)
    flux = np.sin(2 * np.pi * 0.5 * time)
    result = LombScargleAnalyzer(maximum_frequency=1.0, false_alarm_probability=None).analyze(
        time, flux
    )
    assert result.normalization == "standard"
    assert abs(result.dominant_frequency - 0.5) < result.frequency_step * 2
    window = analyze_spectral_window(time, result.frequency)
    assert window.spectral_window_power.shape == result.frequency.shape
    assert np.all(window.spectral_window_power >= 0)
    assert window.symmetry["window_symmetry_max_absolute_error"] < 1e-12
