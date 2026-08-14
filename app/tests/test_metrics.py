import numpy as np
import pytest

from app.evaluation.metrics import (
    pairwise_similarities,
    similarity_statistics,
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