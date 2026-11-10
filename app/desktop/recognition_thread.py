"""Background worker that runs the face pipeline off the Qt UI thread.

Loading SCRFD + ArcFace takes a noticeable amount of time, and every frame
after that still needs detection, embedding, and a Qdrant lookup. None of
that can happen on the UI thread without freezing the window, so this
QThread owns the camera, the embedder, and the database client, and only
ever talks back to the UI through signals.

The one bit of state shared between threads is the pending registration
name: the UI thread writes it when "Register" is clicked, the worker thread
reads it once per frame. Even a single string needs a lock here - without
one, a write from the UI thread could interleave with a read on the worker
thread and corrupt the reference. QMutex makes that access sequential.
"""

from PySide6.QtCore import QMutex, QMutexLocker, QThread, Signal

from app.camera.camera import Camera
from app.config import CameraConfig, get_settings
from app.database.qdrant import FaceDatabase, FaceMatch
from app.desktop.rendering import DetectedFace
from app.embedding.face_embedder import FaceEmbedder
from app.recognition.matching import get_face_label, is_match


class RecognitionThread(QThread):
    """Captures frames and emits annotated results until stop() is called."""

    frame_ready = Signal(object, object)  # (np.ndarray frame, list[DetectedFace])
    status_message = Signal(str)
    error = Signal(str)

    def __init__(self, camera_config: CameraConfig | None = None) -> None:
        super().__init__()
        self._camera_config = camera_config or get_settings().camera
        self._running = False
        self._mutex = QMutex()
        self._pending_name: str | None = None

    def request_registration(self, name: str) -> None:
        """Queue a name to register against the next detected face. UI-thread call."""

        with QMutexLocker(self._mutex):
            self._pending_name = name

    def stop(self) -> None:
        """Ask the loop to exit; caller should still join with wait()."""

        self._running = False

    def run(self) -> None:
        try:
            camera = Camera(self._camera_config)
        except RuntimeError as exc:
            self.error.emit(str(exc))
            return

        try:
            embedder = FaceEmbedder()
            database = FaceDatabase()
        except Exception as exc:  # model files missing, Qdrant unreachable, etc.
            self.error.emit(str(exc))
            camera.release()
            return

        self._running = True
        self.status_message.emit("Recognition running.")

        try:
            while self._running:
                frame = camera.read()
                faces = embedder.embed(frame)

                detected: list[DetectedFace] = []
                first_embedding = None

                for face in faces:
                    embedding = face.normed_embedding

                    if first_embedding is None:
                        first_embedding = embedding

                    match: FaceMatch | None = database.find_match(embedding)
                    label = get_face_label(match)

                    x1, y1, x2, y2 = face.bbox.astype(int)
                    detected.append(
                        DetectedFace(
                            bbox=(x1, y1, x2, y2),
                            label=label,
                            is_match=is_match(match),
                            embedding=embedding,
                        )
                    )

                with QMutexLocker(self._mutex):
                    pending_name = self._pending_name
                    self._pending_name = None

                if pending_name is not None:
                    if first_embedding is None:
                        self.status_message.emit(
                            "No face detected, registration skipped."
                        )
                    else:
                        database.register(pending_name, first_embedding)
                        self.status_message.emit(f"Registered '{pending_name}'.")

                self.frame_ready.emit(frame, detected)
        finally:
            camera.release()
