# Data

## Expected observations

The feature workflow accepts one light curve per astronomical object. A CSV must contain `time` and `flux`; `flux_error` is optional. Values must be numeric and finite after an explicitly documented cleaning policy. Time and flux arrays must have equal length, timestamps must provide a usable baseline, and duplicate or unordered timestamps must be resolved consistently before spectral analysis.

Each object is the independent experimental unit. Cadence samples from one object must not be split across reference and evaluation partitions. Stable object identifiers and aligned row ordering are required when supplying the seven representation matrices to the Level 3 results workflow.

## Provenance and validation

Users should record the original provider, product identifier, retrieval or processing version, object identifier, time and flux semantics, units, uncertainty semantics, and any filtering or transformation. Validation rejects malformed inputs; it does not infer missing provenance or determine redistribution rights. Public requirements are machine-readable in [`configs/data_requirements.json`](../configs/data_requirements.json).

Level 3 additionally needs the frozen operational context used for retained peaks, reference-frequency selection, spectral stability inputs, and typed graph relationships. These scientific decisions are explicit rather than reconstructed heuristically.

## Distribution boundary

The observational datasets used in the scientific study are not redistributed with this repository. To preserve source provenance and avoid redistributing externally sourced observational files, the public repository does not bundle the approximately 2.4 GB publication corpus. Source data must be obtained from the original providers where permitted. This is a repository distribution decision, not a legal determination that redistribution is forbidden.

The known public inventory identifies the study population as the Konkoly K2
Campaign 2 RR Lyrae pilot population. Complete publication-ready bibliographic
entries are not present in the public artifact inventory and remain a
manuscript task. The Level 2 artifact workflow does not require the raw corpus.

## Synthetic demonstration data

The repository includes [one project-created synthetic light curve](../examples/data/synthetic_demo_light_curve.csv) for the Level 1 Quick Start. It is deterministic, normalized, dimensionless, and explicitly not observational data or a publication-corpus member. Its [provenance record](../examples/data/synthetic_demo_provenance.json) and [generator](../examples/generate_synthetic_demo.py) document every generation parameter. It supports software demonstration only and is not publication evidence.

## Frozen publication artifacts

The frozen numerical registry, six main figures, two main tables, and five
supplementary tables are bundled under [`publication/`](../publication/).
They support Level 2 reproduction without including or reconstructing the
observational publication corpus.
