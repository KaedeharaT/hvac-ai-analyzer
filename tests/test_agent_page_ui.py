import os
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from building_ai.config import Settings
from building_ai.i18n import LanguageManager
from building_ai.memory import MemoryStore
from building_ai.observability import TraceStore
from building_ai.ui.context import ApplicationContext
from building_ai.knowledge import KnowledgeService
from building_ai.knowledge_catalog import load_materialized_facts, source_registry
from building_ai.ui.agent_chat import AgentEvidenceCard, AgentProcessCard, AgentSourcesCard
from building_ai.ui.main_window import MainWindow
from building_ai.ui.pages.pages import AgentPage
from building_ai.ui.pages.knowledge_page import KnowledgeBasePage

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


def test_knowledge_page_uses_real_catalog_summary_search_filters_and_i18n(qapp, configured_context):
    KnowledgeService(configured_context.database).ingest_catalog(source_registry(), load_materialized_facts())
    LanguageManager.instance().set_language("en_US")
    page = KnowledgeBasePage(configured_context); page.show()
    assert page.cards["knowledge_trusted_sources"].value.text() == "19"
    assert page.cards["knowledge_chunks"].value.text() == "154"
    assert page.status.text() == "Knowledge Ready"
    assert page.source_table.rowCount() == 19
    page._toggle_source_browser()
    assert page.source_browser.isVisible()
    page.source_table.selectRow(0)
    assert "URL:" in page.source_details.toPlainText()

    page.query.setText("冷冻水温差低")
    page.search()
    assert page._results and page._filtered_results()
    page.country_filter.setCurrentIndex(page.country_filter.findData("US"))
    assert all(item["metadata"]["country"] == "US" for item in page._filtered_results())
    page.country_filter.setCurrentIndex(page.country_filter.findData("all"))
    page.query.setText("既存建物の省エネ改修")
    page.search()
    assert any(item["metadata"]["country"] == "Japan" for item in page._filtered_results())

    LanguageManager.instance().set_language("zh_CN")
    assert page.search_heading.text() == "搜索专业知识"
    assert page.search_button.text() == "搜索知识"
    LanguageManager.instance().set_language("en_US")
    page.close()


def test_knowledge_navigation_uses_its_own_subtitle_without_duplicate_page_heading(qapp, configured_context):
    LanguageManager.instance().set_language("en_US")
    window = MainWindow(configured_context)
    index = next(i for i, (_, key, _) in enumerate(window.NAVIGATION) if key == "knowledge_base")
    window.change_page(index)
    page = window.pages[index]
    assert window.page_title.text() == "Knowledge Base"
    assert "HVAC operation" in window.page_subtitle.text()
    assert not hasattr(page, "heading")
    window.close()


@pytest.mark.parametrize("width,height,columns", [(860, 720, 2), (1048, 720, 4), (1208, 900, 4), (1688, 1080, 4)])
def test_knowledge_page_metrics_reflow_without_text_clipping(qapp, configured_context, width, height, columns):
    KnowledgeService(configured_context.database).ingest_catalog(source_registry(), load_materialized_facts())
    page = KnowledgeBasePage(configured_context); page.resize(width, height); page.show(); QTest.qWait(30)
    assert page.query.isVisible() and page.query.width() > page.search_button.width() * 3
    for card in page.cards.values():
        assert card.value.height() >= card.value.fontMetrics().height()
        assert card.title.height() >= card.title.fontMetrics().height()
        assert card.height() >= card.minimumHeight()
    assert page.metric_grid.columnCount() == columns
    page.close()


def test_citation_card_is_separate_from_project_evidence_and_keeps_ids_technical(qapp):
    source = {
        "chunk_id": "catalog:flow-context", "title": "Flow context", "section": "Engineering Principles", "score": 8.25,
        "metadata": {"organization": "U.S. DOE / NREL", "country": "US", "language": "English", "source_id": "us_energyplus", "official_url": "https://energyplus.net/"},
    }
    citations = AgentSourcesCard([source], {"query": "low delta-T", "trace_id": "trace-1"})
    evidence = AgentEvidenceCard(["AHP-3-3 · COP: 3.94", "Average ΔT: 5.75 °C"])
    assert citations.title.text() == "Reference material"
    assert "U.S. DOE / NREL" in citations.source_labels[0].text()
    assert "catalog:flow-context" not in citations.source_labels[0].text()
    assert any(button.property("citation_open_button") is True for button in citations.findChildren(type(citations.button)))
    citations.button.click()
    assert "catalog:flow-context" in citations.details.toPlainText()
    assert evidence.title.text() == "Project evidence"
    citations.close(); evidence.close()


def test_agent_page_hides_reference_card_when_runtime_has_no_rag_sources(qapp, configured_context):
    page = AgentPage(configured_context)
    trace_id = "no-rag-source"
    TraceStore(configured_context.database).save({
        "trace_id": trace_id, "query": "What equipment is in this project?", "intent": "project_summary",
        "tool_calls": [], "plan": [], "evidence_checks": ["SUFFICIENT"], "reflections": [],
        "memory_used": [], "knowledge_sources": [], "llm_calls": [], "grounded": True, "status": "SUCCEEDED",
    })
    page._request_started = time.monotonic()
    page._agent_completed(SimpleNamespace(trace_id=trace_id, answer="Equipment summary", grounded=True, sources=[]))
    assert not any(isinstance(item, AgentSourcesCard) for item in page.transcript.items)
    page.close()
