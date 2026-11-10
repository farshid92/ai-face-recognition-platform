import cv2

from app.camera.camera import Camera
from app.config import get_settings
from app.detection.face_detector import FaceDetector


def main() -> None:
    camera = Camera(get_settings().camera)
    detector = FaceDetector()

    try:
        while True:
            frame = camera.read()

            faces = detector.detect(frame)

            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                confidence = float(face.det_score)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Face: {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow(
                "AI Face Recognition Platform",
                frame,
            )

            if cv2.waitKey(1) == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()