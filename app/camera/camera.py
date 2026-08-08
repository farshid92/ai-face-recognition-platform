import cv2

class Camera:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap =cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"cannot open camera {camera_index}"
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