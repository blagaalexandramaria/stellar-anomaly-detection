# Synthetic Quick Start

This directory contains one deterministic synthetic light curve created solely to demonstrate the public software APIs. It is not observational data, is not a publication object, does not reproduce any published measurements, and provides no evidence for the scientific claims or candidate rankings.

The signal contains mildly irregular sampling, one sinusoidal component, a weaker second harmonic, and modest seeded Gaussian noise. Flux and uncertainty are normalized and dimensionless; time uses an arbitrary unit. Full parameters and provenance are in [`data/synthetic_demo_provenance.json`](data/synthetic_demo_provenance.json).

## Run the demo

Install the package from the repository root, then run:

```bash
python -m pip install .
python examples/quickstart.py
```

The script loads the bundled CSV and uses the installed public package to compute Lomb–Scargle output, spectral peaks, all 42 PISD features, harmonic graph features, Graph-Core/Extended/All vectors, and the seven representation views. It does not train IF, LOF, or AE because one object is not an anomaly-detection corpus.

Expected high-level output reports the sample and frequency counts, dominant
frequency, retained peaks, typed graph edges, PISD dimension, graph dimensions,
and seven representation views. It ends with `Demo completed successfully`;
it does not print or assign an anomaly label.

## Regenerate the data

```bash
python examples/generate_synthetic_demo.py
```

Generation is deterministic with seed `20260821`. Running the generator twice produces identical CSV and provenance files.

## Reproducibility boundary

This demo verifies installation and the single-object feature/representation path. It does not reproduce publication findings. Use the Level 2 commands for frozen publication figures, tables, and numerical results. Full Level 3 scientific recomputation requires observational data obtained independently from the original sources plus the documented operational context.
