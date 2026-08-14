import numpy as np
from insightface.app import FaceAnalysis
from insightface.app.common import Face


class FaceEmbedder:
    """Generate face embeddings using InsightFace."""

    def __init__(self) -> None:
        self.app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection","recognition"],
            providers=["CPUExecutionProvider"],
        )

        self.app.prepare(ctx_id=0)

    def embed(self, frame: np.ndarray) -> list[Face]:
        """Detect faces and return faces containing embeddings."""
        return self.app.get(frame)