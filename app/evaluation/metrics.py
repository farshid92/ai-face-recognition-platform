from itertools import combinations

import numpy as np

from app.utils.similarity import cosine_similarity


def pairwise_similarities(
    embeddings: np.ndarray,
) -> np.ndarray:
    """Calculate cosine similarity for every unique embedding pair."""

    scores = [
        cosine_similarity(a, b)
        for a, b in combinations(embeddings, 2)
    ]

    return np.asarray(scores, dtype=np.float32)


def similarity_statistics(
    scores: np.ndarray,
) -> dict[str, float]:
    """Calculate descriptive statistics for similarity scores."""

    return {
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores)),
    }