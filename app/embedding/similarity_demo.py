from pathlib import Path

import cv2
import numpy as np

from app.embedding.face_embedder import FaceEmbedder
from app.evaluation.metrics import (
    pairwise_similarities,
    similarity_statistics,
)


def main() -> None:
    embedder = FaceEmbedder()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    embeddings: list[np.ndarray] = []
    target_embeddings = 10

    print(f"Capture {target_embeddings} embeddings.")
    print("Press SPACE to capture.")
    print("Press Q to quit.")

    try:
        while len(embeddings) < target_embeddings:
            success, frame = camera.read()

            if not success:
                break

            faces = embedder.embed(frame)

            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

            cv2.putText(
                frame,
                f"Captured: {len(embeddings)}/{target_embeddings}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow(
                "Embedding Similarity Experiment",
                frame,
            )

            key = cv2.waitKey(1)

            if key == ord(" "):
                if not faces:
                    print("No face detected.")
                    continue

                embedding = faces[0].normed_embedding.copy()

                embeddings.append(embedding)

                print(
                    f"Captured embedding {len(embeddings)}"
                )

            elif key == ord("q"):
                break
            
                

    finally:
        camera.release()
        cv2.destroyAllWindows()

    if len(embeddings) < target_embeddings:
        print(
            f"Only captured {len(embeddings)} "
            f"of {target_embeddings} embeddings."
        )
        return

    output_dir = Path("data/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir/"person_a.npy", np.asarray(embeddings))

    print(f"saved embeddings to {output_dir / 'person_a.npy'}")

    scores = pairwise_similarities(
        np.asarray(embeddings)
    )

    stats = similarity_statistics(scores)

    print()
    print("Same-person similarity experiment")
    print("----------------------------------")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Number of pairs: {len(scores)}")
    print(f"Minimum similarity: {stats['min']:.4f}")
    print(f"Maximum similarity: {stats['max']:.4f}")
    print(f"Mean similarity: {stats['mean']:.4f}")
    print(f"Median similarity: {stats['median']:.4f}")
    print(f"Standard deviation: {stats['std']:.4f}")


if __name__ == "__main__":
    main()