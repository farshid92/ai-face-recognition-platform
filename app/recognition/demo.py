import cv2

from app.database.qdrant import FaceDatabase, FaceMatch
from app.embedding.face_embedder import FaceEmbedder

MATCH_THRESHOLD = 0.50


def get_face_label(match: FaceMatch | None) -> str:
    if match is None or match.score < MATCH_THRESHOLD:
        return "Unknown"
    return match.name


def main() -> None:
    embedder = FaceEmbedder()
    database = FaceDatabase()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Press R to register your face.")
    print("Press Q to quit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                break

            faces = embedder.embed(frame)

            for face in faces:
                embedding = face.normed_embedding

                match = database.find_match(embedding)
                label = get_face_label(match)

                x1, y1, x2, y2 = face.bbox.astype(int)

                if match is not None and match.score >= MATCH_THRESHOLD:
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            cv2.imshow(
                "Face Recognition",
                frame,
            )

            key = cv2.waitKey(1)

            if key == ord("r"):
                if not faces:
                    print("No face detected.")
                    continue

                embedding = faces[0].normed_embedding

                point_id = database.register(
                    "Farshid",
                    embedding,
                )

                print(
                    f"Face registered. ID: {point_id}"
                )

            elif key == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()