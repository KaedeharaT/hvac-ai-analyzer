"""Capture privacy-safe README screenshots from the current production UI.

Every displayed project value is generated from an in-memory synthetic BEMS
dataset and synthetic drawing. No local user project, model path, API key, or
private experiment artifact is read by this script.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt5.QtCore import QEventLoop, QTimer
from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import QApplication

from building_ai.agent_runtime import AgentRuntime
from building_ai.i18n import LanguageManager
from building_ai.knowledge import KnowledgeService
from building_ai.knowledge_catalog import CATALOG_DIR, load_materialized_facts, source_registry
from building_ai.ui.agent_chat import AgentProcessCard
from building_ai.ui.main_window import MainWindow
from scripts.capture_product_qa import _context


OUTPUT = ROOT / "docs" / "images"
PAGES = {
    "dashboard.png": "dashboard",
    "equipment.png": "equipment",
    "energy-analysis.png": "energy_analysis",
    "diagnostics.png": "analysis",
    "knowledge-base.png": "knowledge_base",
    "drawing-intelligence.png": "drawing_intelligence",
}


def process_events(app: QApplication, milliseconds: int = 200) -> None:
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec_()
    app.processEvents()


def configure_font(app: QApplication) -> None:
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
    OUTPUT.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="buildingai-readme-") as value:
        context = _context(Path(value))
        KnowledgeService(context.database).ingest_catalog(
            source_registry(), load_materialized_facts(CATALOG_DIR)
        )
        window = MainWindow(context)
        window.resize(1440, 900)
        window.show()
        process_events(app)

        for filename, page_key in PAGES.items():
            payload = {"equipment_id": context.equipment[0].name} if page_key == "equipment" else {}
            window.navigate_to(page_key, payload)
            process_events(app)
            if page_key == "analysis":
                window.page_by_key["analysis"].tabs.setCurrentIndex(2)
                process_events(app)
            if not window.grab().save(str(OUTPUT / filename), "PNG"):
                raise RuntimeError(f"Could not save {filename}")

        window.resize(1600, 1080)
        process_events(app)
        window.navigate_to("agent", {"equipment_id": context.equipment[0].name})
        agent_page = window.page_by_key["agent"]
        query = "What should I improve first for AHP-01?"
        runtime = AgentRuntime(context)
        route = runtime.route(query)
        plan = runtime.plan(route, context.current_project.project_id)
        agent_page.append_user_message(query)
        agent_page._current_process = AgentProcessCard(route, plan)
        agent_page.transcript.append(agent_page._current_process)
        agent_page._request_started = time.monotonic()
        response = runtime.run(query, context.current_project.project_id, agent_page._conversation_id)
        agent_page._agent_completed(response)
        process_events(app, 500)
        chat_bar = agent_page.chat_scroll.verticalScrollBar()
        chat_bar.setValue(round(chat_bar.maximum() * 0.18))
        process_events(app)
        if not window.grab().save(str(OUTPUT / "ai-assistant.png"), "PNG"):
            raise RuntimeError("Could not save ai-assistant.png")

        window.close()

    print(f"Captured {len(PAGES) + 1} synthetic, privacy-safe README screenshots in {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
