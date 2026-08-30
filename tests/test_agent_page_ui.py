import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from building_ai.config import Settings
from building_ai.i18n import LanguageManager
from building_ai.memory import MemoryStore
from building_ai.ui.context import ApplicationContext
from building_ai.ui.agent_chat import AgentProcessCard
from building_ai.ui.main_window import MainWindow
from building_ai.ui.pages.pages import AgentPage

_APP: QApplication | None = None


class _Provider:
    display_name = "Qwen"
    is_configured = True

    @staticmethod
    def test_connection(timeout=0.35):
        return True, "connected"


class _Manager:
    @staticmethod
    def get_provider():
        return _Provider()


@pytest.fixture(scope="module")
def qapp():
    # Keep a strong module reference: deleting QApplication also deletes Qt
    # singletons that existing non-UI i18n tests use later in the same process.
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


@pytest.fixture
def configured_context(tmp_path):
    context = ApplicationContext(Settings(provider="not_configured", data_dir=tmp_path / "data"))
    context.llm_manager = _Manager()
    return context


def test_agent_page_unconfigured_disables_chat(qapp, tmp_path):
    page = AgentPage(ApplicationContext(Settings(provider="not_configured", data_dir=tmp_path / "data")))
    page.refresh()

    assert not page.input.isEnabled()
    assert not page.send_button.isEnabled()
    page.close()


def test_agent_page_enter_shift_enter_and_backend_failure_is_safe(qapp, configured_context):
    LanguageManager.instance().set_language("en_US")
    page = AgentPage(configured_context)
    submitted = []
    page.message_submitted.connect(submitted.append)
    page.show()

    page.input.setPlainText("Why is COP unavailable?")
    QTest.keyClick(page.input, Qt.Key_Return)
    assert submitted == ["Why is COP unavailable?"]
    assert page._user_message_count == 1
    QTest.qWait(150)
    assert any(getattr(item, "role", "") == "assistant" for item in page.transcript.items)
    assert not any(getattr(item, "translation_key", "") == "agent_backend_not_connected" for item in page.transcript.items)

    page.input.setPlainText("first line")
    QTest.keyClick(page.input, Qt.Key_Return, Qt.ShiftModifier)
    assert submitted == ["Why is COP unavailable?"]
    assert "\n" in page.input.toPlainText()

    page.input.clear()
    page.submit_message()
    assert submitted == ["Why is COP unavailable?"]
    page.close()


def test_agent_page_tool_call_and_dynamic_translation(qapp, configured_context):
    page = AgentPage(configured_context)
    LanguageManager.instance().set_language("en_US")
    page.refresh()
    handle = page.show_tool_call("get_semantic_mapping", "RUNNING")
    page.update_tool_call(handle, "SUCCESS")
    assert "Completed" in page._tool_calls[handle].label.text()
    failed = page.show_tool_call("get_point_timeseries", "FAILED")
    assert "Failed" in page._tool_calls[failed].label.text()

    LanguageManager.instance().set_language("zh_CN")
    assert page.heading.text() == "AI 助手"
    assert page.input.placeholderText() == "输入你的问题……"
    assert "查询完成" in page._tool_calls[handle].label.text()
    assert "查询失败" in page._tool_calls[failed].label.text()
    LanguageManager.instance().set_language("en_US")
    page.close()


def test_agent_page_open_settings_navigation(qapp, configured_context):
    window = MainWindow(configured_context)
    agent_index = next(i for i, (_, key, _) in enumerate(window.NAVIGATION) if key == "agent")
    settings_index = next(i for i, (_, key, _) in enumerate(window.NAVIGATION) if key == "settings")
    page = window.pages[agent_index]
    window.change_page(agent_index)
    page.open_settings()
    assert window.stack.currentIndex() == settings_index
    window.close()


def test_agent_focus_is_conversation_scoped_and_cleared_on_project_switch(qapp, configured_context):
    first = configured_context.projects.create("Project 7")
    second = configured_context.projects.create("Project 8")
    configured_context.open_project(first.project_id)
    page = AgentPage(configured_context); page.refresh()
    MemoryStore(configured_context.database).put(first.project_id, page._conversation_id, "focus", "equipment", {"equipment_id": "AHP-3-3"})
    page._refresh_focus()
    assert "AHP-3-3" in page.focus_status.text()

    configured_context.open_project(second.project_id)
    page.refresh()
    assert page.focus_status.text() == ""
    assert page._conversation_id.startswith(f"gui-{second.project_id}")
    page.close()


def test_agent_process_card_presents_trace_without_internal_tool_names(qapp):
    LanguageManager.instance().set_language("en_US")
    card = AgentProcessCard()
    card.complete({"tool_calls": [{"tool": "get_equipment_kpis", "success": True}, {"tool": "search_knowledge", "success": True}],
                   "knowledge_sources": [{"title": "Guide"}], "reflections": [{"status": "PARTIAL"}], "intent": "equipment_analysis",
                   "plan": [], "evidence_checks": ["PARTIAL", "SUFFICIENT"], "memory_used": [], "llm_calls": []}, 1200)
    assert "data tools" in card.summary.text()
    assert "get_equipment_kpis" not in card.steps.text()
    LanguageManager.instance().set_language("zh_CN")
    assert "分析完成" in card.summary.text()
    assert "get_equipment_kpis" not in card.steps.text()
    LanguageManager.instance().set_language("en_US")
    card.close()
