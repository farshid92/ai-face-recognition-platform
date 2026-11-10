import pytest

from app.config import CameraConfig, Settings


def test_defaults_match_local_development():
    settings = Settings.from_env({})

    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.match_threshold == 0.50
    assert settings.model_name == "buffalo_l"
    assert settings.onnx_providers == ("CPUExecutionProvider",)
    assert settings.cors_origins == ()
    assert settings.camera == CameraConfig()


def test_environment_overrides_defaults():
    settings = Settings.from_env(
        {
            "QDRANT_URL": "http://qdrant:6333",
            "MATCH_THRESHOLD": "0.42",
            "INSIGHTFACE_MODEL": "buffalo_s",
            "ONNX_PROVIDERS": "CUDAExecutionProvider, CPUExecutionProvider",
            "CORS_ORIGINS": "http://localhost:5173,",
            "CAMERA_INDEX": "1",
            "CAMERA_WIDTH": "640",
            "CAMERA_HEIGHT": "480",
        }
    )

    assert settings.qdrant_url == "http://qdrant:6333"
    assert settings.match_threshold == pytest.approx(0.42)
    assert settings.model_name == "buffalo_s"
    assert settings.onnx_providers == (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    assert settings.cors_origins == ("http://localhost:5173",)
    assert settings.camera == CameraConfig(index=1, width=640, height=480)


def test_blank_numeric_values_fall_back_to_defaults():
    settings = Settings.from_env({"MATCH_THRESHOLD": "", "CAMERA_INDEX": " "})

    assert settings.match_threshold == 0.50
    assert settings.camera.index == 0


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("MATCH_THRESHOLD", "high", "MATCH_THRESHOLD must be a number"),
        ("MATCH_THRESHOLD", "1.5", "between -1 and 1"),
        ("CAMERA_INDEX", "front", "CAMERA_INDEX must be an integer"),
        ("ONNX_PROVIDERS", " , ", "at least one provider"),
    ],
)
def test_invalid_values_are_rejected(key: str, value: str, message: str):
    with pytest.raises(ValueError, match=message):
        Settings.from_env({key: value})
