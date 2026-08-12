import logging

import cv2
from camera.camera import Camera
from config import CameraConfig
from utils.fps import FPSCounter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    config = CameraConfig()

    logger.info(
        "Starting camera %s",
        config.index,
    )

    camera = Camera(config)
    fps_counter = FPSCounter()

    try:
        while True:
            frame = camera.read()

            fps = fps_counter.update()

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow(
                "AI Face Recognition Platform",
                frame,
            )

            key = cv2.waitKey(1)

            if key == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

        logger.info("Camera stopped.")


if __name__ == "__main__":
    main()