import numpy as np


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """Calculate cosine similarity between two vectors."""

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        raise ValueError("Cannot compare zero vectors.")

    return float(
        np.dot(a, b) / (a_norm * b_norm)
    )