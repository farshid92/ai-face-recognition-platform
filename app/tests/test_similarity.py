import numpy as np
import pytest

from app.utils.similarity import cosine_similarity


def test_identical_vectors_have_similarity_one():
    vector = np.array([1.0, 2.0, 3.0])

    result = cosine_similarity(vector, vector)

    assert result == pytest.approx(1.0)


def test_opposite_vectors_have_similarity_minus_one():
    vector_a = np.array([1.0, 0.0])
    vector_b = np.array([-1.0, 0.0])

    result = cosine_similarity(vector_a, vector_b)

    assert result == pytest.approx(-1.0)


def test_orthogonal_vectors_have_similarity_zero():
    vector_a = np.array([1.0, 0.0])
    vector_b = np.array([0.0, 1.0])

    result = cosine_similarity(vector_a, vector_b)

    assert result == pytest.approx(0.0)


def test_zero_vector_is_rejected():
    vector_a = np.array([0.0, 0.0])
    vector_b = np.array([1.0, 0.0])

    with pytest.raises(ValueError):
        cosine_similarity(vector_a, vector_b)