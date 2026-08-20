"""FastAPI service for face registration and verification (Module 8)."""

from dataclasses import dataclass
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.database.qdrant import FaceDatabase
from app.embedding.face_embedder import FaceEmbedder
from app.recognition.demo import is_match_score


class HealthResponse(BaseModel):
    status: str


class RegistrationResponse(BaseModel):
    point_id: str
    name: str


class VerificationResponse(BaseModel):
    name: str | None
    score: float | None = Field(default=None, ge=-1.0, le=1.0)
    is_match: bool


@dataclass
class RecognitionService:
    """Coordinate image decoding, embedding, and vector database operations."""

    embedder: FaceEmbedder
    database: FaceDatabase

    @classmethod
    def create(cls) -> "RecognitionService":
        return cls(embedder=FaceEmbedder(), database=FaceDatabase())

    def _embedding_from_image(self, image_bytes: bytes) -> np.ndarray:
        image = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError("The uploaded file is not a readable image.")

        faces = self.embedder.embed(image)

        if len(faces) != 1:
            raise ValueError(
                f"Expected exactly one face, but detected {len(faces)}."
            )

        return faces[0].normed_embedding

    def register(self, name: str, image_bytes: bytes) -> RegistrationResponse:
        embedding = self._embedding_from_image(image_bytes)
        point_id = self.database.register(name, embedding)

        return RegistrationResponse(point_id=point_id, name=name)

    def verify(self, image_bytes: bytes) -> VerificationResponse:
        embedding = self._embedding_from_image(image_bytes)
        match = self.database.find_match(embedding)

        if match is None:
            return VerificationResponse(
                name=None,
                score=None,
                is_match=False,
            )

        is_match = is_match_score(match.score)

        return VerificationResponse(
            name=match.name if is_match else None,
            score=match.score,
            is_match=is_match,
        )


app = FastAPI(
    title="AI Face Recognition Platform",
    version="0.1.0",
    description="Register and verify faces using SCRFD, ArcFace, and Qdrant.",
)
_service: RecognitionService | None = None


def get_service() -> RecognitionService:
    """Create the expensive recognition components only when first needed."""

    global _service

    if _service is None:
        _service = RecognitionService.create()

    return _service


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/api/v1/faces", response_model=RegistrationResponse, status_code=201)
def register_face(
    name: Annotated[str, Form(min_length=1, max_length=100)],
    image: Annotated[UploadFile, File()],
) -> RegistrationResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file.")

    try:
        return get_service().register(name.strip(), image.file.read())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/verify", response_model=VerificationResponse)
def verify_face(
    image: Annotated[UploadFile, File()],
) -> VerificationResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file.")

    try:
        return get_service().verify(image.file.read())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
