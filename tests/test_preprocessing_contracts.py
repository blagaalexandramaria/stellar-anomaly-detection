import numpy as np

from stellar_anomaly_detection.representations.preprocessing import (
    PreprocessingPolicy,
    fit_transform_reference_evaluation,
)


def _fixture():
    reference = np.asarray([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0], [5.0, 4.0]])
    evaluation = np.asarray([[5.0, 5.0], [9.0, 6.0]])
    return reference, evaluation


def test_standard_zero_scale_is_retained_and_zero_for_both_partitions():
    reference, evaluation = _fixture()
    ref, eva = fit_transform_reference_evaluation(
        reference, evaluation, PreprocessingPolicy.STANDARD_SCALE_V1
    )
    assert ref.shape == reference.shape and eva.shape == evaluation.shape
    assert np.array_equal(ref[:, 0], np.zeros(4))
    assert np.array_equal(eva[:, 0], np.zeros(2))
    expected_ref = (reference[:, 1] - reference[:, 1].mean()) / reference[:, 1].std()
    expected_eva = (evaluation[:, 1] - reference[:, 1].mean()) / reference[:, 1].std()
    assert np.allclose(ref[:, 1], expected_ref)
    assert np.allclose(eva[:, 1], expected_eva)


def test_robust_zero_scale_is_retained_and_zero_for_both_partitions():
    reference, evaluation = _fixture()
    ref, eva = fit_transform_reference_evaluation(
        reference, evaluation, PreprocessingPolicy.ROBUST_SCALE_V1
    )
    assert ref.shape[1] == eva.shape[1] == 2
    assert np.array_equal(ref[:, 0], np.zeros(4))
    assert np.array_equal(eva[:, 0], np.zeros(2))
    center = np.median(reference[:, 1])
    scale = np.quantile(reference[:, 1], .75) - np.quantile(reference[:, 1], .25)
    assert np.allclose(ref[:, 1], (reference[:, 1] - center) / scale)
    assert np.allclose(eva[:, 1], (evaluation[:, 1] - center) / scale)


def test_raw_is_identity_and_evaluation_never_affects_reference_fit():
    reference, evaluation = _fixture()
    raw_ref, raw_eva = fit_transform_reference_evaluation(reference, evaluation, "RAW_V1")
    assert np.array_equal(raw_ref, reference) and np.array_equal(raw_eva, evaluation)
    changed_evaluation = evaluation.copy(); changed_evaluation[:, 1] += 1000
    ref_a, _ = fit_transform_reference_evaluation(reference, evaluation, "STANDARD_SCALE_V1")
    ref_b, _ = fit_transform_reference_evaluation(reference, changed_evaluation, "STANDARD_SCALE_V1")
    assert np.array_equal(ref_a, ref_b)
