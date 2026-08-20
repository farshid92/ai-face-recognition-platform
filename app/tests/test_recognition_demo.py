from app.database.qdrant import FaceMatch
from app.recognition.demo import get_face_label


def test_unknown_match_uses_unknown_label():
    assert get_face_label(None) == "Unknown"


def test_low_score_match_is_treated_as_unknown():
    match = FaceMatch(name="Farshid", score=0.49)

    assert get_face_label(match) == "Unknown"


def test_known_match_uses_registered_name():
    match = FaceMatch(name="Farshid", score=0.92)

    assert get_face_label(match) == "Farshid"


def test_threshold_boundary_is_treated_as_match():
    match = FaceMatch(name="Farshid", score=0.50)

    assert get_face_label(match) == "Farshid"
