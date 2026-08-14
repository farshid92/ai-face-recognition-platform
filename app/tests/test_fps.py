import pytest

from app.utils.fps import FPSCounter


def test_fps_counter_starts_at_zero():
    counter = FPSCounter()

    assert counter.update() == 0.0


def test_fps_counter_rejects_invalid_window_size():
    with pytest.raises(ValueError):
        FPSCounter(window_size=0)


def test_fps_counter_returns_positive_value():
    counter = FPSCounter()

    counter.update()
    counter.update()

    assert counter.update() >= 0.0