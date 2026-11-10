# syntax=docker/dockerfile:1

# FastAPI face recognition service (SCRFD + ArcFace via InsightFace).
#
# Build:  docker build -t face-recognition-api .
# The InsightFace model pack is downloaded at build time and baked into the
# image, so containers start without network access to GitHub.

ARG PYTHON_IMAGE=python:3.13-slim-trixie

FROM ghcr.io/astral-sh/uv:0.12.2 AS uv

FROM ${PYTHON_IMAGE} AS runtime

# opencv-python links against libGL and GLib even when no window is opened.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /srv

# Install dependencies first so code changes don't invalidate this layer.
# uv is mounted only for this step, so it doesn't end up in the image.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups

USER app

# Download the model pack into ~/.insightface during the build, then keep
# only the detection and recognition models the app loads. The zip and the
# unused landmark/gender-age models would add ~450 MB to the image.
ARG INSIGHTFACE_MODEL=buffalo_l
ENV INSIGHTFACE_MODEL=${INSIGHTFACE_MODEL}
RUN <<EOF
python - <<'PY'
import os
from pathlib import Path

from insightface.app import FaceAnalysis

face_app = FaceAnalysis(
    name=os.environ["INSIGHTFACE_MODEL"],
    allowed_modules=["detection", "recognition"],
    providers=["CPUExecutionProvider"],
)
face_app.prepare(ctx_id=0)

keep = {Path(model.model_file).resolve() for model in face_app.models.values()}
models_dir = Path.home() / ".insightface" / "models"

for path in models_dir.rglob("*"):
    if path.is_file() and path.resolve() not in keep:
        path.unlink()

print("Kept models:", sorted(p.name for p in keep))
PY
EOF

COPY --chown=app:app app ./app

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

# One worker: each worker loads its own copy of the models into memory.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
