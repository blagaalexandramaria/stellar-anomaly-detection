import numpy as np

from stellar_anomaly_detection.analysis import normalized_rank, normalized_ranks, ordinal_ranks
from stellar_anomaly_detection.pisd.descriptor import feature_order
from stellar_anomaly_detection.representations import (
    REPRESENTATION_IDS,
    RepresentationBuilder,
    representation_registry,
)


def test_pisd_and_representation_contracts():
    assert len(feature_order()) == 42
    assert len(set(feature_order())) == 42
    assert len(REPRESENTATION_IDS) == 7
    views = RepresentationBuilder().build(np.zeros(42), np.zeros(30), np.zeros(36))
    assert {key: value.size for key, value in views.items()} == dict(
        zip(REPRESENTATION_IDS, (42, 30, 36, 66, 72, 78, 108), strict=True)
    )
    assert all(np.isfinite(value).all() for value in views.values())
    assert representation_registry()["PISD_PLUS_GRAPH_ALL"]["analysis_status"] == "PRIMARY"


def test_rank_orientation_and_normalization():
    ranks = ordinal_ranks([0.1, 0.9, 0.2], ["a", "b", "c"])
    assert ranks.tolist() == [3, 1, 2]
    assert normalized_rank(1, 3) == 1.0
    assert normalized_rank(3, 3) == 0.0
    assert normalized_ranks(ranks).tolist() == [0.0, 1.0, 0.5]
    assert normalized_rank(1, 1) == 1.0
