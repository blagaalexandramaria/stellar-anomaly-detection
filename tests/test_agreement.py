import numpy as np

from stellar_anomaly_detection.analysis import kendall_w, pairwise_rank_agreement, top_k_overlap


def test_known_perfect_agreement_fixture():
    ranks = np.arange(1, 11)
    result = pairwise_rank_agreement(ranks, ranks)
    assert np.isclose(result["spearman"], 1.0)
    assert np.isclose(result["kendall"], 1.0)
    assert kendall_w(np.vstack([ranks, ranks, ranks])) == 1.0
    assert [
        row["intersection_count"]
        for row in top_k_overlap({"IF": ranks, "LOF": ranks, "AE": ranks})
    ] == [3, 5, 10]
