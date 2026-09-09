"""Stable spectral-analysis API."""

from .lomb_scargle import LombScargleAnalyzer, LombScargleResult
from .window import (
    SamplingDiagnosticConfiguration,
    SpectralWindowResult,
    analyze_spectral_window,
    compute_spectral_window_power,
)

__all__ = [
    "LombScargleAnalyzer",
    "LombScargleResult",
    "SamplingDiagnosticConfiguration",
    "SpectralWindowResult",
    "analyze_spectral_window",
    "compute_spectral_window_power",
]
