"""Descriptor-level public API for the frozen 42-feature PISD."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stellar_anomaly_detection.pisd.components.spectral_complexity import SpectralComplexity
from stellar_anomaly_detection.pisd.components.spectral_morphology import SpectralMorphology
from stellar_anomaly_detection.pisd.components.spectral_strength import SpectralStrength
from stellar_anomaly_detection.registry import publication_registry


def feature_order() -> tuple[str, ...]:
    """Return all 42 PISD feature identifiers in frozen global order."""
    registry = publication_registry("pisd_feature_registry.json")
    features = sorted(registry["features"], key=lambda value: value["global_order"])
    order = tuple(value["feature_id"] for value in features)
    if registry["feature_count"] != 42 or len(order) != 42 or len(set(order)) != 42:
        raise ValueError("Invalid frozen PISD feature registry")
    return order


def family_feature_order() -> dict[str, tuple[str, ...]]:
    """Return frozen feature ordering grouped by PISD family."""
    registry = publication_registry("pisd_feature_registry.json")
    output: dict[str, list[tuple[int, str]]] = {}
    for feature in registry["features"]:
        output.setdefault(feature["family_id"], []).append(
            (feature["order_within_family"], feature["feature_id"])
        )
    return {name: tuple(value for _, value in sorted(items)) for name, items in output.items()}


class PhysicsInformedSpectralDescriptor:
    """Assemble the frozen 42-feature descriptor in authoritative order.

    Spectral strength, morphology, and complexity are calculated directly from
    a standard-normalized Lomb–Scargle spectrum. Harmonic and stability values
    must be supplied from the frozen operational-reference and Monte Carlo
    policies; the public API does not silently substitute a different method.
    """

    def __init__(self) -> None:
        self.strength = SpectralStrength()
        self.morphology = SpectralMorphology()
        self.complexity = SpectralComplexity()

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return the frozen descriptor feature names."""
        return feature_order()

    def transform(
        self,
        frequency: ArrayLike,
        power: ArrayLike,
        *,
        time_baseline: float,
        harmonic_features: Mapping[str, float | int],
        stability_features: Mapping[str, float | int],
    ) -> NDArray[np.float64]:
        """Return one finite 42-vector without changing frozen semantics."""
        families = family_feature_order()
        calculated = OrderedDict()
        calculated.update(self.strength.extract(frequency, power))
        calculated.update(self.morphology.extract(frequency, power, time_baseline))
        calculated.update(harmonic_features)
        calculated.update(self.complexity.extract(frequency, power))
        calculated.update(stability_features)
        for family in ("harmonic_structure", "spectral_stability"):
            supplied = tuple(name for name in calculated if name in set(families[family]))
            if supplied != families[family]:
                raise ValueError(f"{family} features do not match frozen ordering")
        if tuple(calculated) != self.feature_names:
            raise ValueError("PISD component feature order does not match frozen registry")
        vector = np.asarray(
            [float(calculated[name]) for name in self.feature_names],
            dtype=np.float64,
        )
        if vector.shape != (42,) or not np.all(np.isfinite(vector)):
            raise ValueError("PISD descriptor must be a finite 42-vector")
        vector.setflags(write=False)
        return vector

    def transform_mapping(self, **kwargs: object) -> OrderedDict[str, float]:
        """Return one descriptor as an ordered feature-to-value mapping."""
        vector = self.transform(**kwargs)
        return OrderedDict(zip(self.feature_names, map(float, vector), strict=True))
