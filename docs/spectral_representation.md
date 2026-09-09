# Spectral representation

## Lomb–Scargle analysis

Lomb–Scargle analysis maps irregularly sampled light curves to frequency and power while retaining time-sampling diagnostics. The frozen technical configuration uses `standard` normalization. The public `LombScargleAnalyzer` returns the spectrum; spectral-window utilities expose sampling structure separately.

```python
from stellar_anomaly_detection.spectral import LombScargleAnalyzer
```

## Peaks and reference-frequency semantics

Retained spectral peaks are selected peaks used for descriptor and graph construction. A reference frequency anchors relational interpretation; it is not automatically equivalent to every strongest peak. The public full workflow treats peak retention and reference selection as supplied frozen operational context because changing them would change scientific semantics.

Typed relationships include:

- harmonic and subharmonic relations to the reference frequency;
- rational frequency ratios;
- combination relations formed from reference components;
- sidebands or offsets around a reference component.

Tolerance and precedence rules belong to the frozen implementation/context. These relations encode spectral interpretation, not confirmed physical mode identification.

For the frozen publication results, the operational reference was selected deterministically under `canonical_harmonic_family_lexicographic_v1`. “Operational” is essential: selection supplies a reproducible anchor for PISD and graph interpretation and does not itself confirm an astrophysical fundamental.

## Physics-Informed Spectral Descriptor

PISD is a 42-feature, ordered summary of the spectrum and its interpreted structure. Its five frozen families are:

1. **Spectral Strength** — 5 features describing power magnitude and concentration.
2. **Spectral Morphology** — 13 features describing retained-peak shape and distribution.
3. **Harmonic Structure** — 10 features describing reference-relative organization.
4. **Spectral Complexity** — 7 features describing distributional and relational complexity.
5. **Spectral Stability** — 7 features describing consistency across the frozen stability inputs.

```python
from stellar_anomaly_detection.pisd import (
    PhysicsInformedSpectralDescriptor,
    family_feature_order,
    feature_order,
)
```

The complete names, ordering, types, nullability, and required semantics are maintained in the [supplementary PISD feature registry](../publication/supplementary/registries/pisd_feature_registry.json); duplicating all 42 definitions here would create a second authority.

The frozen identifiers `fundamental_frequency`, `fundamental_power`, and `fundamental_dominance` are retained for schema compatibility. They mean frequency, power, and dominance relative to the operational canonical reference selected by the policy above, not independent physical confirmation. A [source-path crosswalk](../publication/provenance/pisd_source_path_crosswalk.csv) distinguishes historical provenance labels from current public module locations.

Frozen Spectral Stability provenance establishes `N_perturb = 100`, root seed 42, policy `pisd_stability_object_seed_sha256_v1`, 33 deterministic object-specific streams, and fixed-grid reuse within each perturbation run. This compact statement does not redistribute per-object seeds or observational inputs.
