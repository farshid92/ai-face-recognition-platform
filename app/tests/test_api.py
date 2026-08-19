from types import SimpleNamespace

import numpy as np
import pytest

from app.api import RecognitionService, health


class FakeEmbedder:
    def __init__(self, faces: list[object]) -> None:
        self.faces = faces

    def embed(self, image: np.ndarray) -> list[object]:
        return self.faces


class FakeDatabase:
    def __init__(self, match: object | None = None) -> None:
        self.match = match
        self.registered: list[tuple[str, np.ndarray]] = []

    def register(self, name: str, embedding: np.ndarray) -> str:
        self.registered.append((name, embedding))
        return "test-point-id"

    def find_match(self, embedding: np.ndarray) -> object | None:
        return self.match


def _jpeg_bytes() -> bytes:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    success, encoded = __import__("cv2").imencode(".jpg", image)
    assert success
    return encoded.tobytes()


def _service(match: object | None = None) -> tuple[RecognitionService, FakeDatabase]:
    face = SimpleNamespace(normed_embedding=np.ones(512, dtype=np.float32))
    database = FakeDatabase(match)
    return RecognitionService(FakeEmbedder([face]), database), database


def test_health_returns_ok():
    assert health().status == "ok"


def test_register_requires_one_face_and_returns_point_id():
    service, database = _service()

    result = service.register("Farshid", _jpeg_bytes())

    assert result.point_id == "test-point-id"
    assert result.name == "Farshid"
    assert database.registered[0][0] == "Farshid"


def test_invalid_image_is_rejected():
    service, _ = _service()

    with pytest.raises(ValueError, match="readable image"):
        service.verify(b"not an image")


def test_verification_returns_unknown_below_threshold():
    match = SimpleNamespace(name="Farshid", score=0.49)
    service, _ = _service(match)

    result = service.verify(_jpeg_bytes())

    assert result.name is None
    assert result.score == pytest.approx(0.49)
    assert result.is_match is False


def test_verification_returns_name_above_threshold():
    match = SimpleNamespace(name="Farshid", score=0.91)
    service, _ = _service(match)

    result = service.verify(_jpeg_bytes())

    assert result.name == "Farshid"
    assert result.score == pytest.approx(0.91)
    assert result.is_match is True


def test_multiple_faces_are_rejected():
    first = SimpleNamespace(normed_embedding=np.ones(512, dtype=np.float32))
    second = SimpleNamespace(normed_embedding=np.ones(512, dtype=np.float32))
    service = RecognitionService(
        FakeEmbedder([first, second]),
        FakeDatabase(),
    )

    with pytest.raises(ValueError, match="exactly one face"):
        service.verify(_jpeg_bytes())
