"""Capture reviewed, repository-local screenshots for the project README.

This is intentionally a presentation helper, not a test fixture.  It opens the
existing local Project 7 data in the real desktop application and captures the
same pages shown to users.  The AI Assistant capture uses one normal bounded
Agent response so its source cards are genuine; it never changes BEMS data,
semantic mappings, findings, or recommendations.

Run from the repository root:
    python scripts/capture_readme_screenshots.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path


# This lets a headless Windows/Anaconda Qt runtime render the actual UI with a
# normal Windows font instead of replacing labels with blank glyphs.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt5.QtCore import QEventLoop, QTimer
from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import QApplication

from building_ai.i18n import LanguageManager
from building_ai.agent_runtime import AgentRuntime
from building_ai.ui.agent_chat import AgentProcessCard
from building_ai.ui.context import ApplicationContext
from building_ai.ui.main_window import MainWindow


OUTPUT = ROOT / "docs" / "images"


def process_events(app: QApplication, milliseconds: int = 250) -> None:
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec_()
    app.processEvents()


def configure_font(app: QApplication) -> None:
    """Use Segoe UI where available, without changing application production UI."""
    # Microsoft YaHei covers the Chinese language-switch label too; Segoe UI
    # is a good English fallback on minimal Windows systems.
    windows_fonts = Path(os.environ.get("WINDIR", "")) / "Fonts"
    for font_path in (windows_fonts / "msyh.ttc", windows_fonts / "segoeui.ttf"):
        if not font_path.exists():
            continue
        identifier = QFontDatabase.addApplicationFont(str(font_path))
        families = QFontDatabase.applicationFontFamilies(identifier)
        if families:
            app.setFont(QFont(families[0], 10))
            return


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    configure_font(app)
    LanguageManager.instance().set_language("en_US")

    context = ApplicationContext()
    # MainWindow synchronizes the shared language manager from persisted
    # settings, so make this capture explicitly English before it is built.
    context.settings.language = "en_US"
    projects = context.projects.list()
    if not projects:
        raise RuntimeError("A local reviewed project is required to capture README screenshots.")
    project = next((item for item in projects if item.name == "7"), projects[0])
    context.open_project(project.project_id)
    context.ensure_analysis_results()

    window = MainWindow(context)
    window.resize(1440, 920)
    window.show()
    process_events(app)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pages = {
        "dashboard.png": 0,
        "energy-analysis.png": 5,
        "knowledge-base.png": 8,
    }
    for filename, index in pages.items():
        window.change_page(index)
        process_events(app)
        if not window.grab().save(str(OUTPUT / filename), "PNG"):
            raise RuntimeError(f"Could not save {filename}")

    # Capture one real, project-grounded assistant response.  This uses the
    # normal bounded runtime (including project tools and knowledge retrieval)
    # but invokes the existing completion presenter synchronously so the
    # screenshot script does not need a user interaction or a worker thread.
    window.change_page(7)
    agent_page = window.pages[7]
    query = "What should I improve first for AHP-3-3?"
    route = AgentRuntime(context).route(query)
    plan = AgentRuntime(context).plan(route, project.project_id)
    agent_page.append_user_message(query)
    agent_page._current_process = AgentProcessCard(route, plan)
    agent_page.transcript.append(agent_page._current_process)
    agent_page._request_started = time.monotonic()
    response = AgentRuntime(context).run(query, project.project_id, agent_page._conversation_id)
    agent_page._agent_completed(response)
    process_events(app, 500)
    agent_page.chat_scroll.verticalScrollBar().setValue(agent_page.chat_scroll.verticalScrollBar().maximum())
    process_events(app, 100)
    if not window.grab().save(str(OUTPUT / "ai-assistant.png"), "PNG"):
        raise RuntimeError("Could not save ai-assistant.png")

    window.close()
    print(f"Captured {len(pages) + 1} screenshots in {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
