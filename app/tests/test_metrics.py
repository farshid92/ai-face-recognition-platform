import numpy as np
import pytest

from app.evaluation.metrics import (
    equal_error_rate,
    pairwise_similarities,
    roc_curve_scores,
    similarity_statistics,
    verification_accuracy,
)


def test_pairwise_similarity_count():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    scores = pairwise_similarities(embeddings)

    assert len(scores) == 3


def test_pairwise_similarity_values():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )

    scores = pairwise_similarities(embeddings)

    assert scores[0] == pytest.approx(1.0)


def test_similarity_statistics():
    scores = np.array([0.2, 0.4, 0.6, 0.8])

    stats = similarity_statistics(scores)

    assert stats["min"] == pytest.approx(0.2)
    assert stats["max"] == pytest.approx(0.8)
    assert stats["mean"] == pytest.approx(0.5)
    assert stats["median"] == pytest.approx(0.5)


def test_roc_curve_scores_perfect_separation():
    genuine = np.array([0.9, 0.85, 0.95])
    impostor = np.array([0.1, 0.2, 0.05])

    fpr, tpr, thresholds, auc = roc_curve_scores(genuine, impostor)

    assert auc == pytest.approx(1.0)
    assert fpr[0] == pytest.approx(0.0)
    assert tpr[-1] == pytest.approx(1.0)


def test_equal_error_rate_perfect_separation():
    genuine = np.array([0.9, 0.85, 0.95])
    impostor = np.array([0.1, 0.2, 0.05])

    fpr, tpr, thresholds, _ = roc_curve_scores(genuine, impostor)
    eer, threshold = equal_error_rate(fpr, tpr, thresholds)

    assert eer == pytest.approx(0.0)
    assert 0.2 < threshold <= 0.9


def test_verification_accuracy_perfect_separation():
    genuine = np.array([0.9, 0.85, 0.95])
    impostor = np.array([0.1, 0.2, 0.05])

    accuracy = verification_accuracy(genuine, impostor, threshold=0.5)

    assert accuracy == pytest.approx(1.0)