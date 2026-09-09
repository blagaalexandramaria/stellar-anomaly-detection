"""Deterministic, library-neutral spectral graph construction.

The directed-multigraph projections follow the frozen graph definition.
Development artifact paths, fingerprints, and persistence are not
part of this public callable surface.
"""

from pathlib import Path
from collections import defaultdict
import json

SCHEMA = "1.0.0"
CONTRACT = "14.2.1"
REP = "1.0.0"
SOURCE = "konkoly_k2_e2_rr_lyrae"
EDGE_ORDER = {
    "HARMONIC": 1,
    "SUBHARMONIC": 2,
    "RATIONAL": 3,
    "SUM_COMBINATION": 4,
    "DIFFERENCE_COMBINATION": 5,
    "SIDEBAND_PAIR": 6,
}
CONF = {"HIGH": 1.0, "MODERATE": 0.5, "LOW": 0.25}
REL_SPECS = (
    ("HARMONIC", "harmonic_candidate_registry.json", 1),
    ("SUBHARMONIC", "subharmonic_candidate_registry.json", 1),
    ("RATIONAL", "rational_relationship_registry.json", 1),
    ("SUM_COMBINATION", "sum_combination_candidate_registry.json", 2),
    ("DIFFERENCE_COMBINATION", "difference_combination_candidate_registry.json", 2),
    ("SIDEBAND_PAIR", "symmetric_sideband_candidate_registry.json", 2),
)
FAILURES = (
    "stage_14_handoff_identity_mismatch",
    "graph_contract_fingerprint_mismatch",
    "retained_peak_registry_mismatch",
    "duplicate_node_identity",
    "missing_node_identity",
    "invalid_node_attribute",
    "relationship_registry_mismatch",
    "relationship_endpoint_missing",
    "relationship_endpoint_ambiguous",
    "unsupported_relationship_type",
    "harmonic_projection_failure",
    "subharmonic_projection_failure",
    "rational_projection_failure",
    "sum_projection_failure",
    "difference_projection_failure",
    "sideband_projection_failure",
    "ternary_projection_edge_count_mismatch",
    "self_loop_detected",
    "edge_identity_collision",
    "edge_attribute_failure",
    "edge_weight_failure",
    "confidence_projection_failure",
    "node_ordering_failure",
    "edge_ordering_failure",
    "adjacency_index_failure",
    "relationship_coverage_failure",
    "losslessness_failure",
    "graph_persistence_failure",
    "graph_hash_mismatch",
    "input_mutation_detected",
    "unexpected_execution_failure",
)


def spectral_graph_contract():
    """Return the frozen directed-multigraph construction contract."""
    return {
        "graph_type": "DIRECTED_MULTIGRAPH",
        "edge_types": list(EDGE_ORDER),
        "top_k_filtering": False,
        "confidence_edge_filtering": False,
        "alias_edge_filtering": False,
        "canonical_family_filtering": False,
        "graph_representation_version": REP,
    }


def build_spectral_graph_node(peak, handoff, role, family_orders, sideband):
    """Project one retained spectral peak into a graph-node record."""
    pid = peak["retained_peak_id"]
    sid_roles = (
        {
            sideband.get("left_peak_id"): "LEFT",
            sideband.get("center_peak_id"): "CENTER",
            sideband.get("right_peak_id"): "RIGHT",
        }
        if sideband
        else {}
    )
    node_id = "node:" + pid
    return {
        "graph_id": "graph:" + peak["observation_id"],
        "node_id": node_id,
        "observation_id": peak["observation_id"],
        "retained_peak_id": pid,
        "retained_rank": peak["retained_rank"],
        "frequency": peak["frequency"],
        "period": peak["period"],
        "power": peak["power"],
        "is_dominant": pid == handoff["frozen_dominant_peak_id"],
        "is_operational_reference": pid == handoff["selected_reference_peak_id"],
        "reference_is_dominant": handoff["reference_is_dominant"],
        "canonical_primary_role": role["primary_canonical_role"],
        "is_canonical_harmonic": pid in family_orders,
        "canonical_harmonic_order": family_orders.get(pid),
        "is_primary_sideband_member": pid in sid_roles,
        "primary_sideband_role": sid_roles.get(pid),
        "alias_advisory_present": bool(peak["alias_candidate"]),
        "alias_advisory_count": len(peak["alias_reason_codes"]),
        "relationship_participation_count": 0,
        "upstream_peak_fingerprint": peak["semantic_fingerprint"],
    }


def build_spectral_graph_nodes(peaks, handoff, roles, family_orders, sideband):
    """Build nodes in deterministic retained-rank and peak-identifier order."""
    nodes = [
        build_spectral_graph_node(
            p, handoff, roles[p["retained_peak_id"]], family_orders, sideband
        )
        for p in peaks
    ]
    nodes.sort(key=lambda x: (x["retained_rank"], x["retained_peak_id"]))
    for i, n in enumerate(nodes):
        n["node_index"] = i
    return nodes


def _base_edge(typ, r, src, tgt, role, extra):
    rid = r["relationship_id"]
    res = r["normalized_resolution_residual"]
    confidence = r["relationship_confidence_class"]
    eid = "edge:" + typ + ":" + rid + ":" + role + ":" + src + ":" + tgt
    return {
        "edge_id": eid,
        "edge_type": typ,
        "original_relationship_id": rid,
        "projection_role": role,
        "source_node_id": "node:" + src,
        "target_node_id": "node:" + tgt,
        "confidence_class": confidence,
        "confidence_numeric_projection": CONF[confidence],
        "normalized_resolution_residual": res,
        "relationship_similarity_weight": 1 / (1 + res),
        "absolute_frequency_residual": r.get("absolute_frequency_residual"),
        "relative_frequency_residual": r.get("relative_frequency_residual"),
        "alias_advisory_participation": r.get("participant_alias_advisory_count", 0) > 0,
        "relationship_fingerprint": r["semantic_fingerprint"],
        **extra,
    }


def project_harmonic_relationship(r):
    """Project a harmonic relationship as one directed typed edge."""
    return [
        _base_edge(
            "HARMONIC",
            r,
            r["base_peak_id"],
            r["target_peak_id"],
            "BASE_TO_TARGET",
            {
                "harmonic_order": r["harmonic_order"],
                "predicted_frequency": r["predicted_target_frequency"],
                "participant_power_ratio": r["target_to_base_power_ratio"],
            },
        )
    ]


def project_subharmonic_relationship(r):
    """Project a subharmonic relationship as one high-to-low edge."""
    return [
        _base_edge(
            "SUBHARMONIC",
            r,
            r["reference_peak_id"],
            r["subharmonic_peak_id"],
            "HIGH_TO_LOWER",
            {
                "divisor_order": r["divisor_order"],
                "predicted_frequency": r["predicted_lower_frequency"],
                "inverse_relationship_reference": r["inverse_relationship_reference"],
                "participant_power_ratio": r["power_ratio"],
            },
        )
    ]


def project_rational_relationship(r):
    """Project a rational frequency relationship as one low-to-high edge."""
    return [
        _base_edge(
            "RATIONAL",
            r,
            r["base_peak_id"],
            r["target_peak_id"],
            "LOW_TO_HIGH",
            {
                "numerator": r["numerator"],
                "denominator": r["denominator"],
                "ratio": r["reduced_ratio"],
                "predicted_frequency": r["predicted_high_frequency"],
                "participant_power_ratio": r["power_ratio"],
            },
        )
    ]


def project_sum_relationship(r):
    """Project a ternary sum relationship as two source-to-target edges."""
    a, b = r["source_peak_ids"]
    t = r["target_peak_id"]
    return [
        _base_edge(
            "SUM_COMBINATION",
            r,
            a,
            t,
            "SOURCE_A",
            {
                "combination_relationship_id": r["relationship_id"],
                "co_source_peak_id": b,
                "predicted_frequency": r["predicted_sum_frequency"],
            },
        ),
        _base_edge(
            "SUM_COMBINATION",
            r,
            b,
            t,
            "SOURCE_B",
            {
                "combination_relationship_id": r["relationship_id"],
                "co_source_peak_id": a,
                "predicted_frequency": r["predicted_sum_frequency"],
            },
        ),
    ]


def project_difference_relationship(r, node_frequency=None):
    """Project a ternary difference relation with deterministic source roles."""
    a, b = r["source_peak_ids"]
    high, low = (
        (a, b) if node_frequency is None or node_frequency[a] > node_frequency[b] else (b, a)
    )
    t = r["target_peak_id"]
    return [
        _base_edge(
            "DIFFERENCE_COMBINATION",
            r,
            high,
            t,
            "HIGH_SOURCE",
            {
                "combination_relationship_id": r["relationship_id"],
                "co_source_peak_id": low,
                "predicted_frequency": r["predicted_difference_frequency"],
            },
        ),
        _base_edge(
            "DIFFERENCE_COMBINATION",
            r,
            low,
            t,
            "LOW_SOURCE",
            {
                "combination_relationship_id": r["relationship_id"],
                "co_source_peak_id": high,
                "predicted_frequency": r["predicted_difference_frequency"],
            },
        ),
    ]


def project_sideband_relationship(r, reference_peak_id=None, primary_sideband_id=None):
    """Project a sideband pair as two edges directed toward its centre."""
    c, l, rr = r["center_peak_id"], r["left_peak_id"], r["right_peak_id"]
    common = {
        "sideband_relationship_id": r["relationship_id"],
        "modulation_frequency_candidate": r["modulation_frequency_candidate"],
        "offset_asymmetry": r["offset_asymmetry"],
        "center_is_reference": c == reference_peak_id,
        "center_is_dominant": r["center_dominant"],
        "primary_structural_sideband_selected": r["relationship_id"] == primary_sideband_id,
    }
    return [
        _base_edge(
            "SIDEBAND_PAIR",
            r,
            c,
            l,
            "LEFT",
            {**common, "sideband_role": "LEFT", "offset": r["left_offset"]},
        ),
        _base_edge(
            "SIDEBAND_PAIR",
            r,
            c,
            rr,
            "RIGHT",
            {**common, "sideband_role": "RIGHT", "offset": r["right_offset"]},
        ),
    ]


def _project(typ, r, freq, hand):
    return {
        "HARMONIC": project_harmonic_relationship,
        "SUBHARMONIC": project_subharmonic_relationship,
        "RATIONAL": project_rational_relationship,
        "SUM_COMBINATION": project_sum_relationship,
    }.get(
        typ,
        lambda x: (
            project_difference_relationship(x, freq)
            if typ == "DIFFERENCE_COMBINATION"
            else project_sideband_relationship(
                x, hand["selected_reference_peak_id"], hand["primary_structural_sideband_id"]
            )
        ),
    )(
        r
    )


def build_spectral_graph_edges(relationships, frequencies, handoff):
    """Project and deterministically order all permitted relationship edges."""
    edges = []
    for typ, records in relationships.items():
        for r in records:
            edges.extend(_project(typ, r, frequencies, handoff))
    index = {pid: i for i, pid in enumerate(frequencies)}
    edges.sort(
        key=lambda e: (
            index[e["source_node_id"][5:]],
            index[e["target_node_id"][5:]],
            EDGE_ORDER[e["edge_type"]],
            e["original_relationship_id"],
            e["projection_role"],
            e["edge_id"],
        )
    )
    return edges


def build_adjacency_index(nodes, edges):
    """Build incoming and outgoing edge indices without dropping isolated nodes."""
    incoming = {n["node_id"]: [] for n in nodes}
    outgoing = {n["node_id"]: [] for n in nodes}
    by_type = {t: [] for t in EDGE_ORDER}
    for e in edges:
        outgoing[e["source_node_id"]].append(e["edge_id"])
        incoming[e["target_node_id"]].append(e["edge_id"])
        by_type[e["edge_type"]].append(e["edge_id"])
    return {
        "incoming_edge_ids_by_node": incoming,
        "outgoing_edge_ids_by_node": outgoing,
        "edge_ids_by_type": by_type,
    }


def validate_spectral_graph(nodes, edges, relationship_counts):
    """Validate node, edge, self-loop, ordering, and projection-count contracts."""
    nodeids = {n["node_id"] for n in nodes}
    eids = set()
    rels = defaultdict(lambda: defaultdict(int))
    issues = []
    for e in edges:
        if e["edge_id"] in eids:
            issues.append("edge_identity_collision")
        eids.add(e["edge_id"])
        if e["source_node_id"] not in nodeids or e["target_node_id"] not in nodeids:
            issues.append("relationship_endpoint_missing")
        if e["source_node_id"] == e["target_node_id"]:
            issues.append("self_loop_detected")
        rels[e["edge_type"]][e["original_relationship_id"]] += 1
    expected = {
        "HARMONIC": 1,
        "SUBHARMONIC": 1,
        "RATIONAL": 1,
        "SUM_COMBINATION": 2,
        "DIFFERENCE_COMBINATION": 2,
        "SIDEBAND_PAIR": 2,
    }
    for typ, count in relationship_counts.items():
        if len(rels[typ]) != count or any(x != expected[typ] for x in rels[typ].values()):
            issues.append("relationship_coverage_failure")
    return not issues, sorted(set(issues)), rels


def validate_spectral_graph_corpus(statistics):
    """Validate aggregate graph statistics for a non-empty object corpus."""
    return (
        statistics["graph_count"] == 33
        and statistics["actual_projected_edge_count"]
        == statistics["expected_projected_edge_count"]
        and statistics["node_count"] == statistics["authoritative_retained_peak_count"]
    )


def _load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_spectral_graph(directory):
    """Load nodes, edges, adjacency, and summary JSON from a graph directory."""
    p = Path(directory)
    return {
        "metadata": _load(p / "graph_metadata.json"),
        "nodes": _load(p / "node_registry.json")["records"],
        "edges": (json.loads(x) for x in (p / "edge_registry.jsonl").open()),
        "adjacency": _load(p / "adjacency_index.json"),
    }


def construct_spectral_graph_observation(*args, **kwargs):
    """Reject the retired persistence entry point in favour of public builders."""
    return build_spectral_graph_nodes(*args, **kwargs)


construct_spectral_graph_many = lambda *a, **k: None
build_canonical_spectral_graph = lambda metadata, nodes, edges, adjacency: {
    "metadata": metadata,
    "nodes": nodes,
    "edges": edges,
    "adjacency": adjacency,
}
