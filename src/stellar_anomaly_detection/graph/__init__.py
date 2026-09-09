"""Spectral relational graph public API."""

from .construction import (
    build_adjacency_index,
    build_spectral_graph_edges,
    build_spectral_graph_nodes,
    validate_spectral_graph,
)
from .features import (
    extract_graph_all,
    extract_graph_core,
    extract_graph_extended,
    graph_dimensions,
    graph_feature_order,
)

COMMUNITY_PUBLICATION_STATUS = "COMMUNITIES_NOT_SUPPORTED_FOR_PUBLICATION_COLOR_ENCODING"
GRAPH_CORE_DIMENSION = 30
GRAPH_EXTENDED_DIMENSION = 36
GRAPH_ALL_DIMENSION = 66

__all__ = [
    "build_adjacency_index",
    "build_spectral_graph_edges",
    "build_spectral_graph_nodes",
    "validate_spectral_graph",
    "extract_graph_core",
    "extract_graph_extended",
    "extract_graph_all",
    "graph_dimensions",
    "graph_feature_order",
    "GRAPH_CORE_DIMENSION",
    "GRAPH_EXTENDED_DIMENSION",
    "GRAPH_ALL_DIMENSION",
    "COMMUNITY_PUBLICATION_STATUS",
]
