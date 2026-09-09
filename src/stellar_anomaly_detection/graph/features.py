"""Frozen Graph-Core and Graph-Extended feature extraction."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from stellar_anomaly_detection.registry import publication_registry


AUTHORIZED_RELATION_TYPES = frozenset(
    {
        "HARMONIC",
        "SUBHARMONIC",
        "RATIONAL",
        "SUM_COMBINATION",
        "DIFFERENCE_COMBINATION",
        "SIDEBAND_PAIR",
    }
)
RELATIONSHIP_PROJECTION_COUNTS = {
    "HARMONIC": 1,
    "SUBHARMONIC": 1,
    "RATIONAL": 1,
    "SUM_COMBINATION": 2,
    "DIFFERENCE_COMBINATION": 2,
    "SIDEBAND_PAIR": 2,
}


def validate_graph_for_feature_extraction(
    nodes: Sequence[Mapping[str, object]],
    edges: Sequence[Mapping[str, object]],
) -> None:
    """Enforce the frozen directed-multigraph contract at the public boundary.

    Parallel edges and isolated nodes are valid. When frozen relationship IDs
    are supplied, their one- or two-edge projection cardinality is enforced.
    """
    node_ids = [str(node["node_id"]) for node in nodes]
    if not node_ids:
        raise ValueError("invalid_graph_size")
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("duplicate_node_identity")
    node_set = set(node_ids)
    edge_ids: set[str] = set()
    relation_counts: dict[tuple[str, str], int] = defaultdict(int)
    relationship_identity_presence = []
    for edge in edges:
        source = str(edge["source_node_id"])
        target = str(edge["target_node_id"])
        edge_type = str(edge["edge_type"])
        if source not in node_set or target not in node_set:
            raise ValueError("relationship_endpoint_missing")
        if source == target:
            raise ValueError("self_loop_detected")
        if edge_type not in AUTHORIZED_RELATION_TYPES:
            raise ValueError("unsupported_relationship_type")
        if "edge_id" in edge:
            edge_id = str(edge["edge_id"])
            if not edge_id or edge_id in edge_ids:
                raise ValueError("edge_identity_collision")
            edge_ids.add(edge_id)
        has_relationship_id = "original_relationship_id" in edge
        relationship_identity_presence.append(has_relationship_id)
        if has_relationship_id:
            relationship_id = str(edge["original_relationship_id"])
            if not relationship_id:
                raise ValueError("relationship_registry_mismatch")
            relation_counts[(edge_type, relationship_id)] += 1
    if relationship_identity_presence and any(relationship_identity_presence) and not all(
        relationship_identity_presence
    ):
        raise ValueError("relationship_registry_mismatch")
    for (edge_type, _), count in relation_counts.items():
        if count != RELATIONSHIP_PROJECTION_COUNTS[edge_type]:
            raise ValueError("relationship_coverage_failure")


def graph_feature_order(membership: str) -> tuple[str, ...]:
    """Return frozen feature identifiers for Core, Extended, or All membership."""
    registry = publication_registry("graph_feature_registry.json")
    values = [v for v in registry["features"] if v["membership"] == membership]
    return tuple(v["feature_id"] for v in sorted(values, key=lambda v: v["component_order"]))


def graph_dimensions() -> dict[str, int]:
    """Return the frozen Graph-Core, Graph-Extended, and Graph-All dimensions."""
    value = publication_registry("graph_feature_registry.json")["dimensions"]
    expected = {"GRAPH_CORE": 30, "GRAPH_EXTENDED": 36, "GRAPH_ALL": 66}
    if value != expected:
        raise ValueError("Frozen graph dimensions do not match publication contract")
    return dict(value)


def _degrees(node_ids: Sequence[str], edges: Sequence[Mapping[str, object]]):
    index = {node: i for i, node in enumerate(node_ids)}
    incoming = np.zeros(len(node_ids), dtype=np.int64)
    outgoing = np.zeros(len(node_ids), dtype=np.int64)
    unique: set[tuple[str, str]] = set()
    for edge in edges:
        source, target = str(edge["source_node_id"]), str(edge["target_node_id"])
        if source not in index or target not in index:
            raise ValueError("relationship_endpoint_missing")
        outgoing[index[source]] += 1
        incoming[index[target]] += 1
        unique.add((source, target))
    return incoming, outgoing, unique


def _simple_views(node_ids: Sequence[str], unique: set[tuple[str, str]]):
    outgoing = {node: set() for node in node_ids}
    reverse = {node: set() for node in node_ids}
    undirected = {node: set() for node in node_ids}
    for source, target in unique:
        outgoing[source].add(target)
        reverse[target].add(source)
        undirected[source].add(target)
        undirected[target].add(source)
    return outgoing, reverse, undirected


def _weak_components(adjacency: dict[str, set[str]]) -> list[int]:
    seen: set[str] = set()
    sizes: list[int] = []
    for root in sorted(adjacency):
        if root in seen:
            continue
        seen.add(root)
        stack = [root]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in sorted(adjacency[node], reverse=True):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    return sizes


def _strong_components(
    outgoing: dict[str, set[str]], reverse: dict[str, set[str]]
) -> list[int]:
    seen: set[str] = set()
    finish: list[str] = []
    for root in sorted(outgoing):
        if root in seen:
            continue
        seen.add(root)
        stack = [(root, 0, sorted(outgoing[root]))]
        while stack:
            node, index, neighbors = stack[-1]
            if index < len(neighbors):
                neighbor = neighbors[index]
                stack[-1] = (node, index + 1, neighbors)
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append((neighbor, 0, sorted(outgoing[neighbor])))
            else:
                finish.append(node)
                stack.pop()
    seen.clear()
    sizes: list[int] = []
    for root in reversed(finish):
        if root in seen:
            continue
        seen.add(root)
        stack = [root]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in sorted(reverse[node], reverse=True):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    return sizes


def extract_graph_core(
    nodes: Sequence[Mapping[str, object]],
    edges: Iterable[Mapping[str, object]],
) -> tuple[dict[str, float | int], NDArray[np.float64]]:
    """Extract the frozen 30 Graph-Core features from one directed multigraph."""
    nodes = list(nodes)
    edges = list(edges)
    validate_graph_for_feature_extraction(nodes, edges)
    node_ids = [
        str(n["node_id"]) for n in sorted(nodes, key=lambda n: int(n.get("node_index", 0)))
    ]
    references = [str(n["node_id"]) for n in nodes if n["is_operational_reference"]]
    dominants = [str(n["node_id"]) for n in nodes if n["is_dominant"]]
    if len(references) != 1 or len(dominants) != 1:
        raise ValueError("reference_or_dominant_node_resolution_failure")
    incoming, outgoing, unique = _degrees(node_ids, edges)
    n, e, u = len(node_ids), len(edges), len(unique)
    total = incoming + outgoing
    total_mean = float(np.mean(total))
    denominator = n * (n - 1)
    directed, reverse, undirected = _simple_views(node_ids, unique)
    weak, strong = _weak_components(undirected), _strong_components(directed, reverse)
    index = {node: i for i, node in enumerate(node_ids)}

    def role(prefix: str, node: str) -> dict[str, int]:
        i = index[node]
        return {
            f"{prefix}_node_in_degree": int(incoming[i]),
            f"{prefix}_node_out_degree": int(outgoing[i]),
            f"{prefix}_node_total_degree": int(total[i]),
        }

    values: dict[str, float | int] = {
        "node_count": n,
        "projected_edge_count": e,
        "unique_directed_edge_count": u,
        "parallel_edge_count": e - u,
        "parallel_edge_fraction": (e - u) / e if e else 0.0,
        "mean_edge_multiplicity": e / u if u else 0.0,
        "directed_multigraph_density": e / denominator if denominator else 0.0,
        "directed_simple_density": u / denominator if denominator else 0.0,
        "mean_in_degree": float(np.mean(incoming)),
        "mean_out_degree": float(np.mean(outgoing)),
        "mean_total_degree": total_mean,
        "median_total_degree": float(np.median(total)),
        "maximum_in_degree": int(np.max(incoming)),
        "maximum_out_degree": int(np.max(outgoing)),
        "total_degree_std": float(np.std(total, ddof=0)),
        "total_degree_coefficient_of_variation": (
            float(np.std(total, ddof=0) / total_mean) if total_mean else 0.0
        ),
        "in_degree_std": float(np.std(incoming, ddof=0)),
        "out_degree_std": float(np.std(outgoing, ddof=0)),
        "weak_component_count": len(weak),
        "largest_weak_component_fraction": max(weak) / n,
        "strong_component_count": len(strong),
        "largest_strong_component_fraction": max(strong) / n,
        "reciprocal_edge_pair_fraction": (
            sum((b, a) in unique for a, b in unique) / u if u else 0.0
        ),
        "isolated_node_fraction": float(np.sum(total == 0) / n),
        **role("reference", references[0]),
        **role("dominant", dominants[0]),
    }
    order = graph_feature_order("GRAPH_CORE")
    mapping = {name: values[name] for name in order}
    vector = np.asarray([mapping[name] for name in order], dtype=np.float64)
    if vector.shape != (30,) or not np.all(np.isfinite(vector)):
        raise ValueError("graph_core_vector_projection_failure")
    vector.setflags(write=False)
    return mapping, vector


def _gini(values: NDArray[np.float64]) -> float:
    x = np.sort(values)
    total = x.sum()
    n = len(x)
    return (
        0.0
        if total == 0
        else float(2 * np.sum(np.arange(1, n + 1) * x) / (n * total) - (n + 1) / n)
    )


def extract_graph_extended(
    nodes: Sequence[Mapping[str, object]],
    edges: Iterable[Mapping[str, object]],
) -> tuple[dict[str, float | int], NDArray[np.float64]]:
    """Extract the frozen 36 Graph-Extended features."""
    nodes = list(nodes)
    edges = list(edges)
    validate_graph_for_feature_extraction(nodes, edges)
    power = np.asarray([n["power"] for n in nodes], dtype=np.float64)
    frequency = np.asarray([n["frequency"] for n in nodes], dtype=np.float64)
    if (
        np.any(power < 0)
        or not np.all(np.isfinite(power))
        or np.any(frequency <= 0)
        or not np.all(np.isfinite(frequency))
    ):
        raise ValueError("invalid_node_attributes")
    reference = next(n for n in nodes if n["is_operational_reference"])
    dominant = next(n for n in nodes if n["is_dominant"])
    mean, total = float(power.mean()), float(power.sum())
    low, high = float(frequency.min()), float(frequency.max())
    span = high - low
    confidence_projection = {"HIGH": 1.0, "MODERATE": 0.5, "LOW": 0.25}
    type_names = {
        "HARMONIC": "harmonic_edge_fraction",
        "SUBHARMONIC": "subharmonic_edge_fraction",
        "RATIONAL": "rational_edge_fraction",
        "SUM_COMBINATION": "sum_combination_edge_fraction",
        "DIFFERENCE_COMBINATION": "difference_combination_edge_fraction",
        "SIDEBAND_PAIR": "sideband_edge_fraction",
    }
    confidence = defaultdict(int)
    types = defaultdict(int)
    residuals = []
    weights = []
    primary = 0
    for edge in edges:
        c, t = str(edge["confidence_class"]), str(edge["edge_type"])
        if c not in confidence_projection or t not in type_names:
            raise ValueError("invalid_edge_attributes")
        confidence[c] += 1
        types[t] += 1
        residual = edge.get("normalized_resolution_residual")
        weight = edge.get("relationship_similarity_weight")
        if residual is not None:
            residual = float(residual)
            weight = float(weight)
            if residual < 0 or not np.isclose(weight, 1 / (1 + residual), rtol=0, atol=1e-15):
                raise ValueError("similarity_weight_failure")
            residuals.append(residual)
            weights.append(weight)
        if t == "SIDEBAND_PAIR" and edge.get("primary_structural_sideband_selected"):
            primary += 1
    edge_count = len(edges)
    residual_array = np.asarray(residuals, dtype=np.float64)
    harmonic_nodes = [n for n in nodes if n["is_canonical_harmonic"]]
    orders = [n["canonical_harmonic_order"] for n in harmonic_nodes]
    if any(x is None or int(x) < 1 for x in orders):
        raise ValueError("canonical_harmonic_annotation_failure")
    values: dict[str, float | int] = {
        "node_power_mean": mean,
        "node_power_std": float(power.std(ddof=0)),
        "node_power_coefficient_of_variation": (
            float(power.std(ddof=0) / mean) if mean > 0 else 0.0
        ),
        "node_power_gini": _gini(power),
        "reference_node_power_fraction": float(reference["power"]) / total if total else 0.0,
        "dominant_node_power_fraction": float(dominant["power"]) / total if total else 0.0,
        "node_frequency_min": low,
        "node_frequency_max": high,
        "node_frequency_span": span,
        "node_frequency_mean": float(frequency.mean()),
        "node_frequency_std": float(frequency.std(ddof=0)),
        "reference_frequency_relative_position": (
            (float(reference["frequency"]) - low) / span if span else 0.0
        ),
        "dominant_frequency_relative_position": (
            (float(dominant["frequency"]) - low) / span if span else 0.0
        ),
        "reference_to_dominant_frequency_ratio": float(reference["frequency"])
        / float(dominant["frequency"]),
        "reference_to_dominant_power_ratio": float(reference["power"])
        / float(dominant["power"]),
        "high_confidence_edge_fraction": confidence["HIGH"] / edge_count if edge_count else 0.0,
        "moderate_confidence_edge_fraction": (
            confidence["MODERATE"] / edge_count if edge_count else 0.0
        ),
        "low_confidence_edge_fraction": confidence["LOW"] / edge_count if edge_count else 0.0,
        "mean_confidence_projection": (
            sum(confidence_projection[k] * v for k, v in confidence.items()) / edge_count
            if edge_count
            else 0.0
        ),
        **{
            name: types[k] / edge_count if edge_count else 0.0 for k, name in type_names.items()
        },
        "mean_normalized_relationship_residual": (
            float(residual_array.mean()) if len(residual_array) else 0.0
        ),
        "median_normalized_relationship_residual": (
            float(np.median(residual_array)) if len(residual_array) else 0.0
        ),
        "p90_normalized_relationship_residual": (
            float(np.percentile(residual_array, 90, method="linear"))
            if len(residual_array)
            else 0.0
        ),
        "mean_relationship_similarity_weight": float(np.mean(weights)) if weights else 0.0,
        "canonical_harmonic_node_fraction": len(harmonic_nodes) / len(nodes),
        "canonical_harmonic_power_fraction": (
            sum(float(n["power"]) for n in harmonic_nodes) / total if total else 0.0
        ),
        "mean_canonical_harmonic_order": float(np.mean(orders)) if orders else 0.0,
        "maximum_canonical_harmonic_order": int(max(orders)) if orders else 0,
        "alias_advisory_node_fraction": sum(bool(n["alias_advisory_present"]) for n in nodes)
        / len(nodes),
        "primary_sideband_node_fraction": sum(
            bool(n["is_primary_sideband_member"]) for n in nodes
        )
        / len(nodes),
        "primary_sideband_edge_fraction": primary / edge_count if edge_count else 0.0,
    }
    order = graph_feature_order("GRAPH_EXTENDED")
    mapping = {name: values[name] for name in order}
    vector = np.asarray([mapping[name] for name in order], dtype=np.float64)
    if vector.shape != (36,) or not np.all(np.isfinite(vector)):
        raise ValueError("graph_extended_vector_projection_failure")
    vector.setflags(write=False)
    return mapping, vector


def extract_graph_all(
    nodes: Sequence[Mapping[str, object]], edges: Iterable[Mapping[str, object]]
) -> NDArray[np.float64]:
    """Extract the frozen 66-element Graph-All feature vector.

    The output concatenates Graph-Core and Graph-Extended in registry order.
    No analytical relation or isolated node is removed during extraction.
    """
    edges = list(edges)
    _, core = extract_graph_core(nodes, edges)
    _, extended = extract_graph_extended(nodes, edges)
    value = np.concatenate((core, extended))
    value.setflags(write=False)
    if value.shape != (66,):
        raise ValueError("graph_all_vector_projection_failure")
    return value
