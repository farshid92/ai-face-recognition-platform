import time


class FPSCounter:
    """Measure the average frames processed per second."""

    def __init__(self, window_size: int = 30):
        if window_size <= 0:
            raise ValueError("window_size must be greater than zero.")

        self.window_size = window_size
        self._timestamps: list[float] = []

    def update(self) -> float:
        """Register a new frame and return the current FPS."""
        current_time = time.perf_counter()

        self._timestamps.append(current_time)

        if len(self._timestamps) > self.window_size:
            self._timestamps.pop(0)

        if len(self._timestamps) < 2:
            return 0.0

        elapsed = self._timestamps[-1] - self._timestamps[0]

        if elapsed <= 0:
            return 0.0

        return (len(self._timestamps) - 1) / elapsed