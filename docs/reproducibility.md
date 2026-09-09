# Reproducibility

The repository distinguishes synthetic software demonstration, frozen publication-artifact reproduction, and full scientific recomputation. No level turns an anomaly rank into a confirmed astrophysical anomaly.

## Level 1 — Quick Start

Status: `OPERATIONAL_WITH_SYNTHETIC_DEMO`.

Purpose: verify installation and run the public single-object feature and representation APIs on bundled project-created synthetic data.

```bash
python -m pip install .
python examples/quickstart.py
```

The Quick Start loads a 512-sample synthetic light curve, computes Lomb–Scargle output, detects spectral peaks, derives all 42 PISD features with the public harmonic and stability components, extracts Graph-Core/Extended/All features from the demo's harmonic graph, and assembles the seven frozen representation views.

The synthetic signal is not observational data, is not a publication object, and does not reproduce a publication finding. No detector is trained because a single object is not an anomaly-detection corpus. See [the examples guide](../examples/README.md).

## Level 2 — Publication artifact reproduction

Status: `OPERATIONAL`.

Purpose: recover frozen publication numerical results, figures, and tables without raw observational data or model retraining.

```bash
python scripts/reproduce_results.py --mode level2 --output reproduced-results
python scripts/generate_figures.py --mode export --output exported-figures
python scripts/generate_figures.py --mode render --output rendered-figures
```

Expected outputs are the authoritative numerical registry, two main tables, five supplementary tables, and manuscript Figures 1--6. Figure 1 renders from the frozen methodology payload. Figures 2--5 render from frozen public JSON payloads. Figure 6 is artifact-backed: render mode copies the accepted graph-render artifact rather than reconstructing graph topology, layout, or communities. PNGs are exported at 600 dpi.

## Level 3 — Full scientific recomputation

Status: `CONDITIONAL_ON_SOURCE_DATA`.

Purpose: rerun the scientific workflow after independently obtaining the required observational data and operational context.

```bash
python scripts/reproduce_pipeline.py --level 3 \
  --input path/to/light_curve.csv \
  --operational-context path/to/context.json \
  --output full-feature-output
```

The command validates input and computes Lomb–Scargle and spectral-window outputs. With explicit retained-peak, reference-frequency, stability, and relationship context, it writes `spectral/`, `pisd/`, `graph/`, and `representations/` outputs. The raw-to-operational-context interpretation adapter is not turnkey.

Seven aligned representation matrices and object identifiers can be validated and the exact 30 deterministic object-level splits materialized:

```bash
python scripts/reproduce_results.py --mode full \
  --representations path/to/representation_matrices \
  --object-ids path/to/object_ids.txt \
  --output full-results-inputs
```

Automated execution of the complete IF/LOF/AE configuration grid and frozen final aggregation is not wired into `full` mode. Level 3 is therefore not a turnkey publication rerun.

Compact non-sensitive provenance for the frozen selector, stability execution, representation views, experimental design, detector grids, ranking, consensus, and robustness is provided in [`publication/provenance/methodology_provenance.json`](../publication/provenance/methodology_provenance.json). It documents the frozen run; it does not expand the conditional Level 3 execution boundary.

Lomb–Scargle provenance is deliberately qualified. Frozen execution artifacts directly establish the 0.01–20 d⁻¹ range, five samples per peak, Baluev false-alarm method, and alpha 0.01. The current public implementation defaults to `standard` normalization, `fit_mean=True`, `center_data=True`, and `nterms=1`; the compact frozen snapshot did not independently record those four constructor fields. Current defaults are therefore not presented as stronger frozen-execution proof.

## Three distinct data layers

- **Synthetic demo data:** project-created software-usability input; no scientific or anomaly claim.
- **Publication observational data:** externally sourced scientific input; not redistributed here.
- **Frozen publication artifacts:** public numerical results, figures, and tables reproducible through Level 2 without the observational corpus.

## Interpretation and limitations

Rank 1 and normalized rank 1.0 mean most unusual. Preprocessing is fitted on reference objects only. Agreement and robustness describe ordering consistency, not accuracy or physical correctness. Graph structure supplies complementary spectral relational context and does not establish novel phenomena or robust communities.

Open Level 3 limitations are:

- observational source data must be obtained independently;
- explicit operational spectral context is required;
- raw-to-context orchestration is incomplete;
- full detector-grid execution and aggregation are not automated;
- Figure 6 regeneration is artifact-backed rather than semantic-payload-rendered.

These limitations do not affect the operational Level 1 demo or Level 2 artifact reproduction.
