import hashlib

import numpy as np
import pytest

from stellar_anomaly_detection.graph import (
    GRAPH_ALL_DIMENSION,
    GRAPH_CORE_DIMENSION,
    GRAPH_EXTENDED_DIMENSION,
)
from stellar_anomaly_detection.graph.features import (
    extract_graph_all,
    extract_graph_core,
    extract_graph_extended,
    validate_graph_for_feature_extraction,
)


def fixture_graph():
    nodes = [
        {
            "node_id": "node:a",
            "frequency": 1.0,
            "power": 1.0,
            "retained_rank": 1,
            "is_dominant": True,
            "is_operational_reference": True,
            "is_canonical_harmonic": False,
            "canonical_harmonic_order": None,
            "is_primary_sideband_member": False,
            "alias_advisory_present": False,
        },
        {
            "node_id": "node:b",
            "frequency": 2.0,
            "power": 0.5,
            "retained_rank": 2,
            "is_dominant": False,
            "is_operational_reference": False,
            "is_canonical_harmonic": True,
            "canonical_harmonic_order": 2,
            "is_primary_sideband_member": False,
            "alias_advisory_present": False,
        },
    ]
    edges = [
        {
            "source_node_id": "node:a",
            "target_node_id": "node:b",
            "edge_type": "HARMONIC",
            "confidence_class": "HIGH",
            "confidence_numeric_projection": 1.0,
            "normalized_resolution_residual": 0.0,
            "relationship_similarity_weight": 1.0,
            "alias_advisory_participation": False,
        }
    ]
    return nodes, edges


def test_graph_feature_dimensions_and_determinism():
    nodes, edges = fixture_graph()
    core_mapping, core = extract_graph_core(nodes, edges)
    extended_mapping, extended = extract_graph_extended(nodes, edges)
    assert len(core_mapping) == len(core) == GRAPH_CORE_DIMENSION == 30
    assert len(extended_mapping) == len(extended) == GRAPH_EXTENDED_DIMENSION == 36
    assert len(extract_graph_all(nodes, edges)) == GRAPH_ALL_DIMENSION == 66
    assert core_mapping == extract_graph_core(nodes, edges)[0]
    assert hashlib.sha256(core.tobytes()).hexdigest() == (
        "3eea268a00538fa0dbbc6b8665e590b7251c57149939cfc0458cff78f6b19425"
    )
    assert hashlib.sha256(extended.tobytes()).hexdigest() == (
        "6797bc615049d329112aa5affeedc51fc2890f8e92d5c4c0fd0522d677e1d49c"
    )


def test_self_loop_and_unauthorized_relation_are_rejected():
    nodes, edges = fixture_graph()
    loop = [{**edges[0], "target_node_id": "node:a"}]
    with pytest.raises(ValueError, match="self_loop_detected"):
        extract_graph_core(nodes, loop)
    unauthorized = [{**edges[0], "edge_type": "UNAUTHORIZED"}]
    with pytest.raises(ValueError, match="unsupported_relationship_type"):
        extract_graph_extended(nodes, unauthorized)


def test_isolated_nodes_and_legitimate_multiedges_are_preserved():
    nodes, edges = fixture_graph()
    nodes.append(
        {
            **nodes[1], "node_id": "node:c", "frequency": 3.0, "retained_rank": 3,
            "is_canonical_harmonic": False, "canonical_harmonic_order": None,
        }
    )
    edges.append({**edges[0], "edge_type": "RATIONAL"})
    validate_graph_for_feature_extraction(nodes, edges)
    mapping, _ = extract_graph_core(nodes, edges)
    assert mapping["node_count"] == 3
    assert mapping["projected_edge_count"] == 2
    assert mapping["parallel_edge_count"] == 1
    assert np.isclose(mapping["isolated_node_fraction"], 1 / 3)


def test_relationship_projection_count_is_enforced_when_ids_are_supplied():
    nodes, edges = fixture_graph()
    incomplete = [{**edges[0], "edge_type": "SIDEBAND_PAIR", "edge_id": "edge:1",
                   "original_relationship_id": "sideband:1"}]
    with pytest.raises(ValueError, match="relationship_coverage_failure"):
        validate_graph_for_feature_extraction(nodes, incomplete)
