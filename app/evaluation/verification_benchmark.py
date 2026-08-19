"""LFW verification benchmark.

A near-perfect AUC on LFW is expected, not a sign of a groundbreaking model.
LFW has been saturated by deep learning face recognition since around 2018,
and published ArcFace results already sit at 99.6-99.8% on this same split.
The pipeline being tested here (SCRFD detection, ArcFace embedding) is a
production model, not something trained from scratch, so this benchmark
mainly confirms the pipeline is wired correctly end to end. Comparing model
quality today requires harder protocols such as IJB-C or MegaFace, where
scores still spread out enough to tell models apart.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_lfw_pairs

from app.embedding.face_embedder import FaceEmbedder
from app.evaluation.metrics import (
    equal_error_rate,
    roc_curve_scores,
    verification_accuracy,
)
from app.utils.similarity import cosine_similarity

OUTPUT_DIR = Path("data/evaluation")


def _to_bgr_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an LFW float RGB image in [0, 1] to an OpenCV BGR uint8 image."""

    rgb = (image * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def score_pairs(
    embedder: FaceEmbedder,
    pairs: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Detect + embed each LFW pair and split similarity scores by label."""

    genuine_scores: list[float] = []
    impostor_scores: list[float] = []
    skipped = 0

    for (image_a, image_b), label in zip(pairs, labels):
        faces_a = embedder.embed(_to_bgr_uint8(image_a))
        faces_b = embedder.embed(_to_bgr_uint8(image_b))

        if not faces_a or not faces_b:
            skipped += 1
            continue

        score = cosine_similarity(
            faces_a[0].normed_embedding,
            faces_b[0].normed_embedding,
        )

        if label == 1:
            genuine_scores.append(score)
        else:
            impostor_scores.append(score)

    return (
        np.asarray(genuine_scores, dtype=np.float32),
        np.asarray(impostor_scores, dtype=np.float32),
        skipped,
    )


def main() -> None:
    print("Loading LFW verification pairs (downloads on first run)...")

    dataset = fetch_lfw_pairs(subset="test", color=True, resize=1.0, slice_=None)

    embedder = FaceEmbedder()

    genuine, impostor, skipped = score_pairs(
        embedder, dataset.pairs, dataset.target
    )

    print(f"Genuine pairs scored: {len(genuine)}")
    print(f"Impostor pairs scored: {len(impostor)}")
    print(f"Pairs skipped (no face detected): {skipped}")

    fpr, tpr, thresholds, auc = roc_curve_scores(genuine, impostor)
    eer, eer_threshold = equal_error_rate(fpr, tpr, thresholds)
    accuracy = verification_accuracy(genuine, impostor, eer_threshold)

    print()
    print("LFW Face Verification Benchmark")
    print("--------------------------------")
    print(f"AUC: {auc:.4f}")
    print(f"Equal Error Rate: {eer:.4f} at threshold {eer_threshold:.4f}")
    print(f"Accuracy @ EER threshold: {accuracy:.4f}")
    print()
    print(
        "Note: LFW is near-saturated for modern models. These numbers mostly "
        "confirm the detection/embedding pipeline works, not that the model "
        "is state of the art. See the module docstring for details."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez(
        OUTPUT_DIR / "lfw_scores.npz",
        genuine=genuine,
        impostor=impostor,
    )

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Face Verification ROC Curve (LFW)")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "roc_curve.png")

    print(f"Saved ROC curve to {OUTPUT_DIR / 'roc_curve.png'}")


if __name__ == "__main__":
    main()
