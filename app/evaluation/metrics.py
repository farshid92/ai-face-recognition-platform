from itertools import combinations

import numpy as np
from sklearn.metrics import auc as auc_score
from sklearn.metrics import roc_curve

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


def roc_curve_scores(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute the ROC curve and AUC for genuine vs. impostor similarity scores."""

    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate(
        [np.ones_like(genuine_scores), np.zeros_like(impostor_scores)]
    )

    fpr, tpr, thresholds = roc_curve(labels, scores)

    return fpr, tpr, thresholds, float(auc_score(fpr, tpr))


def equal_error_rate(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[float, float]:
    """Find the threshold where the false accept and false reject rates are equal."""

    fnr = 1 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))

    eer = float((fpr[index] + fnr[index]) / 2)

    return eer, float(thresholds[index])


def verification_accuracy(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    threshold: float,
) -> float:
    """Calculate verification accuracy at a fixed similarity threshold."""

    correct = np.sum(genuine_scores >= threshold) + np.sum(
        impostor_scores < threshold
    )
    total = len(genuine_scores) + len(impostor_scores)

    return float(correct / total)