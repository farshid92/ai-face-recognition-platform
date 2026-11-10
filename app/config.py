"""Runtime configuration, read from environment variables.

Every value has a default that matches local development (webcam 0, Qdrant
on localhost, CPU inference), so nothing needs to be set to run the demos.
Deployments override values through the environment, e.g. in Docker:

    QDRANT_URL=http://qdrant:6333
    MATCH_THRESHOLD=0.45
    ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720


@dataclass(frozen=True)
class Settings:
    qdrant_url: str = "http://localhost:6333"
    match_threshold: float = 0.50
    model_name: str = "buffalo_l"
    onnx_providers: tuple[str, ...] = ("CPUExecutionProvider",)
    cors_origins: tuple[str, ...] = ()
    camera: CameraConfig = field(default_factory=CameraConfig)

    def __post_init__(self) -> None:
        if not -1.0 <= self.match_threshold <= 1.0:
            raise ValueError(
                "MATCH_THRESHOLD must be a cosine similarity between -1 and 1, "
                f"got {self.match_threshold}."
            )

        if not self.onnx_providers:
            raise ValueError("ONNX_PROVIDERS must name at least one provider.")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Build settings from environment variables, falling back to defaults."""

        env = os.environ if env is None else env
        defaults = cls()

        return cls(
            qdrant_url=env.get("QDRANT_URL", defaults.qdrant_url),
            match_threshold=_float(env, "MATCH_THRESHOLD", defaults.match_threshold),
            model_name=env.get("INSIGHTFACE_MODEL", defaults.model_name),
            onnx_providers=_csv(env, "ONNX_PROVIDERS", defaults.onnx_providers),
            cors_origins=_csv(env, "CORS_ORIGINS", defaults.cors_origins),
            camera=CameraConfig(
                index=_int(env, "CAMERA_INDEX", defaults.camera.index),
                width=_int(env, "CAMERA_WIDTH", defaults.camera.width),
                height=_int(env, "CAMERA_HEIGHT", defaults.camera.height),
            ),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, read from the environment once."""

    return Settings.from_env()


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}.") from exc


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be a number, got {raw!r}.") from exc


def _csv(
    env: Mapping[str, str],
    key: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    raw = env.get(key)
    if raw is None:
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())
