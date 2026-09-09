"""Frozen ordinal-rank, agreement, consensus, and robustness definitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import kendalltau, rankdata, spearmanr

FROZEN_TOP_K_VALUES = (3, 5, 10)


def ordinal_ranks(
    scores: ArrayLike, object_ids: Sequence[str] | None = None
) -> NDArray[np.int64]:
    """Descending unique ordinal ranks; deterministic object ID breaks score ties."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a finite vector")
    ids = (
        list(object_ids)
        if object_ids is not None
        else [f"row:{i:09d}" for i in range(len(values))]
    )
    if len(ids) != len(values) or len(set(ids)) != len(ids):
        raise ValueError("object_ids must be unique and aligned")
    order = sorted(range(len(values)), key=lambda i: (-float(values[i]), ids[i]))
    ranks = np.empty(len(values), dtype=np.int64)
    for rank, index in enumerate(order, 1):
        ranks[index] = rank
    return ranks


def normalized_rank(rank: int, count: int) -> float:
    """Map ordinal rank to [0, 1], with 1.0 denoting most unusual."""
    if count < 1 or rank < 1 or rank > count:
        raise ValueError("invalid rank")
    return 1.0 if count == 1 else 1.0 - (rank - 1) / (count - 1)


def normalized_ranks(ranks: ArrayLike) -> NDArray[np.float64]:
    """Normalize a complete ordinal-rank permutation."""
    value = np.asarray(ranks, dtype=np.int64)
    if value.ndim != 1 or set(value.tolist()) != set(range(1, len(value) + 1)):
        raise ValueError("ranks must be a complete ordinal permutation")
    return np.asarray(
        [normalized_rank(int(rank), len(value)) for rank in value], dtype=np.float64
    )


def pairwise_rank_agreement(left: ArrayLike, right: ArrayLike) -> dict[str, float]:
    """Return Spearman and Kendall agreement for aligned rank vectors."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if (
        a.shape != b.shape
        or a.ndim != 1
        or len(a) < 2
        or not np.all(np.isfinite(a))
        or not np.all(np.isfinite(b))
    ):
        raise ValueError("invalid rank vectors")
    spearman = spearmanr(a, b)
    kendall = kendalltau(a, b)
    return {
        "spearman": float(spearman.statistic),
        "spearman_p_value": float(spearman.pvalue),
        "kendall": float(kendall.statistic),
        "kendall_p_value": float(kendall.pvalue),
    }


def kendall_w(rank_matrix: ArrayLike) -> float:
    """Return global Kendall's W for model-by-object ranks."""
    matrix = np.asarray(rank_matrix, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] < 2
        or matrix.shape[1] < 2
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("invalid Kendall W input")
    models, objects = matrix.shape
    sums = matrix.sum(axis=0)
    centered = sums - sums.mean()
    s = float(np.sum(centered**2))
    tie_correction = 0.0
    for row in matrix:
        _, counts = np.unique(row, return_counts=True)
        tie_correction += float(np.sum(counts**3 - counts))
    denominator = models**2 * (objects**3 - objects) - models * tie_correction
    if denominator <= 0:
        raise ValueError("undefined Kendall W")
    return 12.0 * s / denominator


def top_k_overlap(
    rankings: Mapping[str, ArrayLike], k_values: Sequence[int] = FROZEN_TOP_K_VALUES
) -> list[dict[str, object]]:
    """Return frozen top-k intersections across aligned rankings."""
    arrays = {name: np.asarray(value, dtype=np.int64) for name, value in rankings.items()}
    if len(arrays) < 2 or len({x.shape for x in arrays.values()}) != 1:
        raise ValueError("rankings must be aligned")
    n = len(next(iter(arrays.values())))
    records = []
    for k in k_values:
        if k not in FROZEN_TOP_K_VALUES or k > n:
            raise ValueError("only frozen k=3,5,10 values are supported")
        sets = {
            name: set(np.flatnonzero(value <= k).tolist()) for name, value in arrays.items()
        }
        if any(len(value) != k for value in sets.values()):
            raise ValueError("top-k cardinality mismatch")
        common = set.intersection(*sets.values())
        records.append(
            {"k": k, "intersection_count": len(common), "intersection_indices": sorted(common)}
        )
    return records


def consensus_ranking(
    normalized_by_model: Mapping[str, ArrayLike],
) -> dict[str, NDArray[np.float64] | NDArray[np.int64]]:
    """Aggregate three normalized-rank vectors by the frozen median rule."""
    matrix = np.vstack(
        [np.asarray(value, dtype=np.float64) for value in normalized_by_model.values()]
    )
    if (
        matrix.ndim != 2
        or matrix.shape[0] != 3
        or not np.all(np.isfinite(matrix))
        or np.any((matrix < 0) | (matrix > 1))
    ):
        raise ValueError("consensus requires three aligned normalized-rank vectors")
    unusualness = np.median(matrix, axis=0)
    ranks = rankdata(-unusualness, method="min").astype(np.int64)
    ranges = np.ptp(matrix, axis=0)
    agreement = 1.0 - ranges
    return {
        "median_normalized_rank": unusualness,
        "consensus_rank": ranks,
        "normalized_rank_range": ranges,
        "cross_model_agreement": agreement,
    }


def ranking_robustness(
    reference_ranks: ArrayLike, comparison_ranks: ArrayLike
) -> dict[str, float | int]:
    """Describe rank agreement and absolute displacement."""
    reference = np.asarray(reference_ranks, dtype=np.float64)
    comparison = np.asarray(comparison_ranks, dtype=np.float64)
    agreement = pairwise_rank_agreement(reference, comparison)
    displacement = np.abs(reference - comparison)
    return {
        **agreement,
        "mean_absolute_rank_displacement": float(np.mean(displacement)),
        "maximum_absolute_rank_displacement": int(np.max(displacement)),
    }
