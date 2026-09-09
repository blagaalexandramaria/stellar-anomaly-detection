"""Compatibility exports for public spectral-graph node construction."""

from stellar_anomaly_detection.graph.construction.spectral_graph_builder import (
    build_spectral_graph_node,
    build_spectral_graph_nodes,
)

__all__ = ["build_spectral_graph_node", "build_spectral_graph_nodes"]
