from app.database.qdrant import FaceMatch
from app.recognition.matching import get_face_label, is_match, is_match_score


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


def test_missing_score_is_never_a_match():
    assert is_match_score(None) is False
    assert is_match(None) is False


def test_explicit_threshold_overrides_configured_one():
    match = FaceMatch(name="Farshid", score=0.60)

    assert is_match(match, threshold=0.70) is False
    assert get_face_label(match, threshold=0.55) == "Farshid"
