import cv2
from config import CameraConfig


class Camera:

    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap =cv2.VideoCapture(config.index)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"cannot open camera {config.index}"
            )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            config.width,
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            config.height,
        )

    def read(self):
        success, frame = self.cap.read()

        if not success:
            raise RuntimeError(
                "Failed to capture frame."                    
            )

        return frame

    def release(self):
        self.cap.release()