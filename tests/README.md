# Tests

The test suite validates the public numerical and reproducibility contracts of
the repository, including spectral processing, PISD stability, graph
construction, preprocessing, detector interfaces, rank agreement, registry
loading, and publication artifact consistency.

Run from the repository root:

```bash
pytest
```

The tests are intended to detect changes that would alter frozen scientific or
publication contracts. They do not recompute the observational study.
