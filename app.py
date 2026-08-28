from __future__ import annotations

import logging
import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from building_ai.ui.main_window import MainWindow
from building_ai.config import Settings


def main() -> int:
    settings = Settings.load()
    log_dir = Path(settings.data_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "building_ai.log", encoding="utf-8")],
    )
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
