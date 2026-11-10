from collections.abc import Sequence

import numpy as np
from insightface.app import FaceAnalysis
from insightface.app.common import Face

from app.config import get_settings


class FaceDetector:
    """Detect faces in images using InsightFace."""

    def __init__(
        self,
        model_name: str | None = None,
        providers: Sequence[str] | None = None,
    ) -> None:
        settings = get_settings()

        self.app = FaceAnalysis(
            name=model_name or settings.model_name,
            allowed_modules=["detection"],
            providers=list(providers or settings.onnx_providers),
        )

        self.app.prepare(ctx_id=0)

    def detect(
        self,
        frame: np.ndarray,
    ) -> list[Face]:
        return self.app.get(frame)
