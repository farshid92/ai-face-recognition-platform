"""Entry point for the desktop application (Module 7)."""

import sys

from PySide6.QtWidgets import QApplication

from app.desktop.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
