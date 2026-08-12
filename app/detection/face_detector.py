import numpy as np
from insightface.app import FaceAnalysis
from insightface.app.common import Face


class FaceDetector:
    """Detect faces in images using InsightFace."""

    def __init__(self) -> None:
        self.app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection"],
            providers=["CPUExecutionProvider"],
        )

        self.app.prepare(ctx_id=0)

    def detect(
        self,
        frame: np.ndarray,
    ) -> list[Face]:
        return self.app.get(frame)