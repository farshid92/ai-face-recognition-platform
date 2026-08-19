import numpy as np

from app.desktop.rendering import DetectedFace, draw_faces, frame_to_qimage


def _face(bbox: tuple[int, int, int, int], is_match: bool) -> DetectedFace:
    return DetectedFace(
        bbox=bbox,
        label="Farshid" if is_match else "Unknown",
        is_match=is_match,
        embedding=np.zeros(512, dtype=np.float32),
    )


def test_draw_faces_does_not_mutate_input_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    draw_faces(frame, [_face((10, 10, 50, 50), is_match=True)])

    assert np.array_equal(frame, np.zeros((100, 100, 3), dtype=np.uint8))


def test_draw_faces_paints_inside_the_bbox():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    annotated = draw_faces(frame, [_face((10, 10, 50, 50), is_match=True)])

    assert annotated[10:50, 10:50].sum() > 0


def test_draw_faces_with_no_faces_returns_unchanged_copy():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    annotated = draw_faces(frame, [])

    assert annotated is not frame
    assert np.array_equal(annotated, frame)


def test_frame_to_qimage_preserves_dimensions():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    image = frame_to_qimage(frame)

    assert image.width() == 200
    assert image.height() == 100
