# Method

## Overview

The pipeline maps each stellar light curve to spectral and graph-derived feature views, applies three unsupervised detector families, and analyzes ranks across representations, configurations, splits, and models. Its output is an ordering for inspection, not a class label.

## Light-curve ingestion

An object is represented by finite time and flux samples, with optional flux uncertainty. The public input boundary is a CSV with `time`, `flux`, and optional `flux_error`. Validation checks matching lengths, finite values, usable timestamp ordering, and a frequency grid meaningful for the observation baseline. Astronomical objects—not individual cadence samples—are the independent units.

## Lomb–Scargle

The Lomb–Scargle periodogram converts irregularly sampled observations to frequency and power arrays. The frozen implementation uses `standard` normalization. Sampling-window diagnostics are retained separately so cadence structure is not silently treated as stellar spectral evidence.

## Spectral interpretation

Retained peaks and an explicit reference frequency support harmonic, subharmonic, rational, combination, and sideband relationships. These are scientifically meaningful operational context. The public Level 3 command requires them explicitly rather than guessing them.

The frozen publication methodology selected that operational reference with the deterministic policy `canonical_harmonic_family_lexicographic_v1`. This records frozen provenance; it does not claim that the public Level 3 command performs turnkey raw-data canonical selection.

## PISD

The Physics-Informed Spectral Descriptor contains 42 ordered features in five families: Spectral Strength, Spectral Morphology, Harmonic Structure, Spectral Complexity, and Spectral Stability. See [Spectral representation](spectral_representation.md) and the [feature registry](../publication/supplementary/registries/pisd_feature_registry.json).

The frozen feature identifiers `fundamental_frequency`, `fundamental_power`, and `fundamental_dominance` are legacy schema names. Here they are computed relative to the operational canonical reference and do not independently establish a physically confirmed stellar fundamental frequency.

## Spectral graph

Retained peaks become nodes in a directed multigraph; typed edges encode permitted frequency relationships. Graph-Core (30 features) describes primary relational structure, Graph-Extended (36) adds complementary topology, and Graph-All concatenates both (66). See [Graph representation](graph_representation.md).

## Representation views

Seven frozen inputs support ablation and combination analysis: `PISD_ONLY`, `GRAPH_CORE_ONLY`, `GRAPH_EXTENDED_ONLY`, `GRAPH_ALL`, `PISD_PLUS_GRAPH_CORE`, `PISD_PLUS_GRAPH_EXTENDED`, and `PISD_PLUS_GRAPH_ALL`. Raw, standard-scaled, and robust-scaled policies are fit only on reference objects. The authoritative ordering and dimensions are in the [representation registry](../publication/supplementary/registries/final_representation_registry.json).

## Unsupervised models

Isolation Forest, novelty-mode Local Outlier Factor, and a symmetric fully connected autoencoder consume each view. Each produces a score oriented so higher is more unusual. Fixed sensitivity grids are aggregated; configurations were not selected against evaluation objects. See [Models](models.md).

## Experimental protocol

The frozen population contains 33 independent astronomical objects. Thirty deterministic object-level repeated-holdout splits use approximately 26 reference and 7 evaluation objects per split. Preprocessing and model fitting use reference objects only. The root seed is 42, with deterministic child seeds derived from experiment identity. Exact rules are in the [experimental protocol registry](../publication/supplementary/registries/experimental_protocol_registry.json).

Frozen PISD stability execution used 100 perturbations for each of 33 objects. Its deterministic object-specific streams use policy `pisd_stability_object_seed_sha256_v1` with root seed 42.

## Ranking and agreement

Rank 1 and normalized rank 1.0 mean most unusual. Consensus uses median normalized rank. Spearman correlation, Kendall correlation, Kendall's W, and predefined top-k intersections describe agreement. They do not measure accuracy.

## Robustness

Frozen perturbations compare aggregation choices and model-specific contributions through rank correlations and displacement. They characterize ranking stability within this dataset and protocol; they do not prove physical anomaly status or generalization to another survey.
