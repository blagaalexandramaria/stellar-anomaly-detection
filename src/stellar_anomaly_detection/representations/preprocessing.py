"""Leakage-safe preprocessing for frozen representation matrices."""

from __future__ import annotations

from enum import Enum
import numpy as np
from numpy.typing import ArrayLike, NDArray


class PreprocessingPolicy(str, Enum):
    """Supported frozen preprocessing policies for detector inputs."""
    RAW_V1 = "RAW_V1"
    STANDARD_SCALE_V1 = "STANDARD_SCALE_V1"
    ROBUST_SCALE_V1 = "ROBUST_SCALE_V1"


def fit_transform_reference_evaluation(
    reference: ArrayLike,
    evaluation: ArrayLike,
    policy: PreprocessingPolicy | str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit preprocessing on reference rows only and apply it to both partitions."""
    reference = np.asarray(reference, dtype=np.float64)
    evaluation = np.asarray(evaluation, dtype=np.float64)
    policy = PreprocessingPolicy(policy)
    if (
        reference.ndim != 2
        or evaluation.ndim != 2
        or reference.shape[1] != evaluation.shape[1]
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(evaluation))
    ):
        raise ValueError("invalid preprocessing matrices")
    if policy is PreprocessingPolicy.RAW_V1:
        return reference.copy(), evaluation.copy()
    if policy is PreprocessingPolicy.STANDARD_SCALE_V1:
        center = reference.mean(axis=0)
        scale = reference.std(axis=0, ddof=0)
    else:
        center = np.median(reference, axis=0)
        scale = np.quantile(reference, 0.75, axis=0, method="linear") - np.quantile(
            reference, 0.25, axis=0, method="linear"
        )
    zero_scale = scale == 0
    reference_output = np.zeros_like(reference, dtype=np.float64)
    evaluation_output = np.zeros_like(evaluation, dtype=np.float64)
    np.divide(reference - center, scale, out=reference_output, where=~zero_scale)
    np.divide(evaluation - center, scale, out=evaluation_output, where=~zero_scale)
    return reference_output, evaluation_output
