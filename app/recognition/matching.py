"""Decide whether a database match is close enough to count as the same person.

This is the single place the match threshold is applied. The webcam demo,
the desktop app, and the API all go through these helpers so they can never
disagree about who is "Unknown".
"""

from app.config import get_settings
from app.database.qdrant import FaceMatch

UNKNOWN_LABEL = "Unknown"


def is_match_score(score: float | None, threshold: float | None = None) -> bool:
    """Return True when score reaches the threshold (MATCH_THRESHOLD by default)."""

    if score is None:
        return False

    if threshold is None:
        threshold = get_settings().match_threshold

    return score >= threshold


def is_match(match: FaceMatch | None, threshold: float | None = None) -> bool:
    return match is not None and is_match_score(match.score, threshold)


def get_face_label(match: FaceMatch | None, threshold: float | None = None) -> str:
    if match is None or not is_match(match, threshold):
        return UNKNOWN_LABEL
    return match.name
