# Stellar Anomaly Detection

A reproducible, physics-informed, multi-view framework for unsupervised ranking of unusual stellar light-curve behaviour. It combines Lomb–Scargle spectral analysis, the 42-feature Physics-Informed Spectral Descriptor (PISD), spectral relational graph features, seven representation views, Isolation Forest (IF), Local Outlier Factor (LOF), and an autoencoder (AE), then studies representation dependence, ranking robustness, and cross-model agreement. Rankings identify inspection candidates; they do not establish confirmed astrophysical anomalies.

## Scientific overview

```text
light curve -> Lomb–Scargle -> PISD / spectral graph -> representation views
            -> IF / LOF / AE -> anomaly ranks -> robustness and agreement
            -> inspection candidates
```

PISD summarizes Spectral Strength, Spectral Morphology, Harmonic Structure,
Spectral Complexity, and Spectral Stability. The graph representation supplies
complementary relational context among retained spectral peaks. Fixed model
and configuration grids are applied to each view under object-level repeated
holdout; scores are oriented so that higher means more unusual, and ranks are
compared without supervised labels.

## Key representations

| Canonical view | Components | Dimensions |
|---|---|---:|
| `PISD_ONLY` | PISD | 42 |
| `GRAPH_CORE_ONLY` | Graph-Core | 30 |
| `GRAPH_EXTENDED_ONLY` | Graph-Extended | 36 |
| `GRAPH_ALL` | Graph-Core + Graph-Extended | 66 |
| `PISD_PLUS_GRAPH_CORE` | PISD + Graph-Core | 72 |
| `PISD_PLUS_GRAPH_EXTENDED` | PISD + Graph-Extended | 78 |
| `PISD_PLUS_GRAPH_ALL` | PISD + Graph-All | 108 |

The frozen definitions and ordering are in [`final_representation_registry.json`](publication/supplementary/registries/final_representation_registry.json).

## Installation

Python 3.11–3.14 is supported by the package metadata.

```bash
git clone https://github.com/blagaalexandramaria/stellar-anomaly-detection.git
cd stellar-anomaly-detection
python -m pip install .
```

Install only the extras needed for a workflow:

```bash
python -m pip install ".[ml]"             # IF and LOF
python -m pip install ".[ae]"             # autoencoder
python -m pip install ".[visualization]"  # figure rendering
python -m pip install ".[dev]"            # tests
```

## Quick start

After installation, run the bundled synthetic demonstration from the repository root:

```bash
python examples/quickstart.py
```

It loads project-created synthetic data and demonstrates Lomb–Scargle analysis, the 42-feature PISD, harmonic spectral graph features, and all seven representation views. It does not use observational data, train detectors, or reproduce publication claims. See [the examples guide](examples/README.md).

Publication-artifact commands remain:

```bash
python scripts/reproduce_results.py --mode level2 --output reproduced-results

python scripts/generate_figures.py --mode list
python scripts/generate_figures.py --mode export --output exported-figures
python scripts/generate_figures.py --mode render --output rendered-figures
```

`reproduce_pipeline.py` remains the user-supplied-data command; Level 3 requires frozen downstream interpretation context. `reproduce_results.py --mode full` validates seven user-supplied representation matrices and object identifiers and materializes the frozen splits, but does not yet automate the detector grid and final aggregation. `generate_figures.py` renders manuscript Figures 2--5 from frozen payloads and preserves Figure 6 from its accepted artifact. See [Reproducibility](docs/reproducibility.md) for exact boundaries.

## Reproducibility levels

- **Level 1 — synthetic Quick Start:** `OPERATIONAL_WITH_SYNTHETIC_DEMO`. The bundled project-created signal verifies installation and the public feature/representation path; it is not publication evidence.
- **Level 2 — publication artifact reproduction:** `OPERATIONAL`. No raw 2.4 GB corpus is required. It covers six final figures, two main tables, five supplementary tables, and the authoritative numerical registry.
- **Level 3 — full scientific recomputation:** `CONDITIONAL_ON_SOURCE_DATA`. It requires user-supplied observations and frozen operational context. Raw-to-context orchestration and automated IF/LOF/AE grid execution and aggregation are not yet fully turnkey.

The frozen publication artifact layer is reproducible without the raw observational corpus; full end-to-end recomputation requires source data and additional operational context.

## Documentation

- [Method](docs/method.md) — scientific workflow and experimental protocol;
- [Data](docs/data.md) — synthetic demo, observational-data boundary, and provenance;
- [Reproducibility](docs/reproducibility.md) — Level 1, Level 2, and Level 3;
- [Spectral representation](docs/spectral_representation.md) — Lomb–Scargle and PISD;
- [Graph representation](docs/graph_representation.md) — graph semantics and features;
- [Models](docs/models.md) — IF, LOF, and AE interfaces;
- [Results](docs/results.md) — repository-facing publication-output summary.

## Data availability

The observational datasets used in the scientific study are **not redistributed** with this repository. To preserve source provenance and avoid redistributing externally sourced files, users must obtain observations from the original providers where permitted and prepare the input described in [Data](docs/data.md) and [`data_requirements.json`](configs/data_requirements.json). The bundled synthetic demo is separate from the publication corpus.

## Publication artifacts

The frozen inventory is documented in [publication/README.md](publication/README.md):

- [`publication/figures/`](publication/figures/) — manuscript Figures 1--6;
- [`publication/tables/`](publication/tables/) — two main tables;
- [`publication/supplementary/`](publication/supplementary/) — five supplementary tables and scientific registries.

## Scientific interpretation boundary

All rankings are unsupervised, and top-ranked objects are inspection candidates.
Rank agreement is not classification accuracy or physical verification. Graph
context is complementary spectral relational information. Graph communities are
not used for publication colour encoding because community assignments are not
treated as a validated scientific output.

## Repository structure

- `src/stellar_anomaly_detection/` — public spectral, PISD, graph, representation, model, and ranking APIs;
- `scripts/` — reproduction and artifact commands;
- `configs/` — reader-facing reproducibility and data requirements;
- `tests/` — public software and contract checks;
- `docs/` — method, data, model, result, and reproducibility guides;
- `publication/` — frozen publication artifacts and registries;
- `examples/` — deterministic synthetic data, its generator, and the Quick Start.

## Citation

Use [`CITATION.cff`](CITATION.cff). No paper or archival DOI has been assigned.

## License

The software and project-created synthetic demo are distributed under the [MIT License](LICENSE).
