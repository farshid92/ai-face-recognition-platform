import cv2
import numpy as np

from app.embedding.face_embedder import FaceEmbedder


def main() -> None:
    embedder = FaceEmbedder()
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    printed = False

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            faces = embedder.embed(frame)

            for face in faces:
                if face.embedding is not None and not printed:
                    raw_embedding = face.embedding
                    normed_embedding = face.normed_embedding

                    print("Raw embedding shape:", raw_embedding.shape)
                    print("Raw embedding norm:", np.linalg.norm(raw_embedding))
                    print("Normalized embedding shape:", normed_embedding.shape)
                    print("Normalized embedding norm:", np.linalg.norm(normed_embedding))
                    print("First 10 raw values:")
                    print(raw_embedding[:10])
                    print("First 10 normalized values:")
                    print(normed_embedding[:10])

                    printed = True

                x1, y1, x2, y2 = face.bbox.astype(int)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

            cv2.imshow(
                "Face Embedding Demo",
                frame,
            )

            if cv2.waitKey(1) == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()