"""Framework-light rendering helpers for the desktop UI.

These functions take plain numpy arrays in and out. Keeping them free of
QThread/QWidget means they can be unit tested without a running Qt event
loop, and reused as-is if frames ever need to be drawn on for a web client.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtGui import QImage

MATCH_COLOR = (0, 255, 0)
UNKNOWN_COLOR = (0, 0, 255)


@dataclass(eq=False)
class DetectedFace:
    """One face detected in a frame, already matched against the database."""

    bbox: tuple[int, int, int, int]
    label: str
    is_match: bool
    embedding: np.ndarray


def draw_faces(frame: np.ndarray, faces: list[DetectedFace]) -> np.ndarray:
    """Draw a box and label per face on a copy of frame; leaves frame untouched."""

    annotated = frame.copy()

    for face in faces:
        x1, y1, x2, y2 = face.bbox
        color = MATCH_COLOR if face.is_match else UNKNOWN_COLOR

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            face.label,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return annotated


def frame_to_qimage(frame: np.ndarray) -> QImage:
    """Convert a BGR OpenCV frame into a QImage a QLabel can display."""

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    bytes_per_line = channels * width

    # .copy() so the QImage owns its buffer once `rgb` goes out of scope.
    return QImage(
        rgb.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888,
    ).copy()
