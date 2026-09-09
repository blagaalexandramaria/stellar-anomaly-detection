import numpy as np
import pytest

pytest.importorskip("sklearn")
from stellar_anomaly_detection.models import IsolationForestDetector, LocalOutlierFactorDetector


def test_reference_fit_and_upward_score_orientation():
    rng = np.random.default_rng(42)
    reference = rng.normal(size=(30, 2))
    evaluation = np.array([[0.0, 0.0], [20.0, 20.0]])
    for model in (IsolationForestDetector(), LocalOutlierFactorDetector(n_neighbors=5)):
        scores = model.fit(reference).score(evaluation)
        assert scores[1] > scores[0]
