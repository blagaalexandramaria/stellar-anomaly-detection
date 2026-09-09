# Results and publication artifacts

This page orients repository users to frozen outputs; it is not a replacement for the manuscript Results section.

## Ranking findings

The seven views produce representation-dependent rankings, which is the intended ablation result rather than evidence that one representation is universally superior. Frozen perturbations quantify ranking robustness across aggregation and model-specific choices. Representative evidence cases include high-consensus inspection candidates and model-sensitivity cases; none is labeled a confirmed astrophysical anomaly.

Across model-level consensus rankings, the frozen agreement statistics are:

| Models | Spearman | Kendall |
|---|---:|---:|
| IF–LOF | 0.8773395721925135 | 0.7121212121212122 |
| IF–AE | 0.8094919786096257 | 0.6174242424242424 |
| LOF–AE | 0.921457219251337 | 0.7386363636363638 |

Global Kendall's W is `0.9129530600118836` over 33 objects. These values quantify rank concordance, not classification accuracy, physical verification, or discovery.

## Spectral relational context

The accepted graph figure supplies relational context for retained peaks and typed spectral relationships. Display geometry is not a physical coordinate system, and community assignments are not a publication result.

## Where to inspect outputs

- [Figures](../publication/figures/) contain the six accepted main figures.
- [Main tables](../publication/tables/) contain two tables in CSV, JSON, and TeX.
- [Supplementary material](../publication/supplementary/) contains five supplementary tables and the feature, representation, model, protocol, and case registries.
- The [authoritative numerical registry](../publication/freeze/registries/final_numerical_results_registry.json) preserves stored precision and interpretation boundaries.

See [Reproducibility](reproducibility.md) for artifact export and rendering commands.
