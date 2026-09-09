# Graph representation

## Construction and semantics

Each retained spectral peak is a node. Directed, typed edges encode allowed harmonic, subharmonic, rational, combination, or sideband relationships. Multiple relationship types may connect the same node pair, so the scientific object is a directed multigraph rather than an untyped proximity network.

The graph captures relational and topological spectral context: connectivity, typed edge patterns, reference-relative organization, and local/global summaries. It does not turn the periodogram into physical stellar coordinates, and node positions in publication figures are visualization geometry only.

```python
from stellar_anomaly_detection.graph import (
    build_spectral_graph_edges,
    build_spectral_graph_nodes,
    extract_graph_all,
    extract_graph_core,
    extract_graph_extended,
)
```

## Feature groups

- **Graph-Core:** 30 primary relational features.
- **Graph-Extended:** 36 complementary topological features.
- **Graph-All:** 66 features formed by concatenating Core and Extended.

The [graph feature registry](../publication/supplementary/registries/graph_feature_registry.json) is the ordering authority. Graph-only and PISD-plus-graph views are defined in the [representation registry](../publication/supplementary/registries/final_representation_registry.json).

## Community-analysis boundary

Diagnostic community analysis did not provide sufficient stability for publication-level community coloring. Diagnostic memberships are therefore not exposed as a scientific result, and no robust astrophysical graph-community claim is made. Publication graph displays communicate spectral relationships and selected local context only.
