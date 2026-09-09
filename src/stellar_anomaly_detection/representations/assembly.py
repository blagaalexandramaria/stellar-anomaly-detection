"""Assemble the seven frozen detector-input views without feature selection."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stellar_anomaly_detection.registry import publication_registry

REPRESENTATION_IDS = (
    "PISD_ONLY",
    "GRAPH_CORE_ONLY",
    "GRAPH_EXTENDED_ONLY",
    "GRAPH_ALL",
    "PISD_PLUS_GRAPH_CORE",
    "PISD_PLUS_GRAPH_EXTENDED",
    "PISD_PLUS_GRAPH_ALL",
)


def representation_registry() -> dict[str, dict[str, object]]:
    """Load and validate the seven frozen representation definitions."""
    value = publication_registry("final_representation_registry.json")
    records = {record["canonical_id"]: record for record in value["records"]}
    if tuple(records) != REPRESENTATION_IDS:
        raise ValueError("Frozen representation identifiers or order changed")
    expected = dict(zip(REPRESENTATION_IDS, (42, 30, 36, 66, 72, 78, 108), strict=True))
    if {key: records[key]["dimensionality"] for key in records} != expected:
        raise ValueError("Frozen representation dimensionalities changed")
    return records


class RepresentationBuilder:
    """Build all frozen views from PISD, Graph-Core, and Graph-Extended arrays."""

    def build(
        self,
        pisd: ArrayLike,
        graph_core: ArrayLike,
        graph_extended: ArrayLike,
    ) -> dict[str, NDArray[np.float64]]:
        """Concatenate validated components into all seven registry-ordered views."""
        values = [np.asarray(x, dtype=np.float64) for x in (pisd, graph_core, graph_extended)]
        if any(x.ndim not in (1, 2) for x in values) or len({x.ndim for x in values}) != 1:
            raise ValueError("Representation components must share rank 1 or 2")
        if values[0].shape[-1] != 42 or values[1].shape[-1] != 30 or values[2].shape[-1] != 36:
            raise ValueError("Frozen component dimensions are 42, 30, and 36")
        if values[0].ndim == 2 and len({x.shape[0] for x in values}) != 1:
            raise ValueError("Representation components must share row count")
        if any(not np.all(np.isfinite(x)) for x in values):
            raise ValueError("Representation components must be finite")
        p, c, e = values
        g = np.concatenate((c, e), axis=-1)
        views = {
            "PISD_ONLY": p,
            "GRAPH_CORE_ONLY": c,
            "GRAPH_EXTENDED_ONLY": e,
            "GRAPH_ALL": g,
            "PISD_PLUS_GRAPH_CORE": np.concatenate((p, c), axis=-1),
            "PISD_PLUS_GRAPH_EXTENDED": np.concatenate((p, e), axis=-1),
            "PISD_PLUS_GRAPH_ALL": np.concatenate((p, g), axis=-1),
        }
        dimensions = {key: value.shape[-1] for key, value in views.items()}
        expected = {
            key: int(value["dimensionality"])
            for key, value in representation_registry().items()
        }
        if tuple(views) != REPRESENTATION_IDS or dimensions != expected:
            raise ValueError("Frozen representation contract mismatch")
        for value in views.values():
            value.setflags(write=False)
        return views
