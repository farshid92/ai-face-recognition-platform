"""Desktop window for the face recognition pipeline (Module 7).

Deliberately thin: it draws frames handed to it and forwards button clicks
to RecognitionThread. All detection/embedding/matching logic already existed
from earlier modules and is untouched here.
"""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.desktop.recognition_thread import RecognitionThread
from app.desktop.rendering import DetectedFace, draw_faces, frame_to_qimage


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI Face Recognition Platform")

        self._thread: RecognitionThread | None = None

        self.video_label = QLabel("Press Start to begin.")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Name to register")

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.register_button = QPushButton("Register Face")
        self.register_button.setEnabled(False)

        self.status_label = QLabel("Idle.")

        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self._stop)
        self.register_button.clicked.connect(self._register)

        controls = QHBoxLayout()
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.name_input)
        controls.addWidget(self.register_button)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addLayout(controls)
        layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def _start(self) -> None:
        self._thread = RecognitionThread()
        self._thread.frame_ready.connect(self._on_frame_ready)
        self._thread.status_message.connect(self.status_label.setText)
        self._thread.error.connect(self._on_error)
        self._thread.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.register_button.setEnabled(True)
        self.status_label.setText("Starting camera and models...")

    def _stop(self) -> None:
        if self._thread is not None:
            self._thread.stop()
            self._thread.wait()
            self._thread = None

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.register_button.setEnabled(False)
        self.status_label.setText("Stopped.")

    def _register(self) -> None:
        name = self.name_input.text().strip()

        if not name:
            self.status_label.setText("Enter a name before registering.")
            return

        if self._thread is not None:
            self._thread.request_registration(name)

    def _on_error(self, message: str) -> None:
        self.status_label.setText(f"Error: {message}")
        self._stop()

    def _on_frame_ready(
        self,
        frame: np.ndarray,
        faces: list[DetectedFace],
    ) -> None:
        annotated = draw_faces(frame, faces)
        image = frame_to_qimage(annotated)
        pixmap = QPixmap.fromImage(image)

        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        self._stop()
        event.accept()
