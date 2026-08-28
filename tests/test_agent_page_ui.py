import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from building_ai.config import Settings
from building_ai.i18n import LanguageManager
from building_ai.ui.context import ApplicationContext
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
