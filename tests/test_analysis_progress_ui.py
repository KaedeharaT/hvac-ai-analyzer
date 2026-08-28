import time

import pandas as pd
import pytest
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from building_ai.config import Settings
from building_ai.i18n import LanguageManager
from building_ai.models import AnalysisResult, SemanticResult
from building_ai.services.analytics_service import AnalyticsService
from building_ai.services.equipment_service import EquipmentService
from building_ai.ui.context import ApplicationContext
from building_ai.ui.pages.pages import AnalyticsPage


_APP: QApplication | None = None


@pytest.fixture(scope="module")
def qapp():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _context(tmp_path):
    context = ApplicationContext(Settings(provider="not_configured", data_dir=tmp_path / "data"))
    project = context.projects.create("Progress")
    context.current_project = project
    context.dataframe = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=8, freq="h"),
        "CH-1 Supply °C": [6.0] * 8,
        "CH-1 Return °C": [8.0] * 8,
        "CH-1 Flow m3/h": [3.6] * 8,
        "CH-1 Power kW": [8.0] * 8,
    })
    points = [
        SemanticResult("CH-1 Supply °C", "heat_source_supply_temp", point_id="s", unit="°C", column="CH-1 Supply °C"),
        SemanticResult("CH-1 Return °C", "heat_source_return_temp", point_id="r", unit="°C", column="CH-1 Return °C"),
        SemanticResult("CH-1 Flow m3/h", "heat_source_flow", point_id="f", unit="m³/h", column="CH-1 Flow m3/h"),
        SemanticResult("CH-1 Power kW", "heat_source_power", point_id="p", unit="kW", column="CH-1 Power kW"),
    ]
    context.semantic_result = AnalysisResult(points, {item.raw_name: item.canonical_label for item in points})
    context.import_metadata = {"time_column": "time"}
    return context


def _wait_for_worker(page):
    for _ in range(100):
        if page._analysis_worker is None:
            return
        QTest.qWait(20)
    raise AssertionError("Analysis worker did not finish")


def test_analysis_worker_reports_real_stages_and_completion(qapp, tmp_path):
    page = AnalyticsPage(_context(tmp_path)); page.show()
    page.run_diagnosis()
    assert page._analysis_worker is not None
    assert page.progress_panel.isVisible()
    assert not page.run_button.isEnabled()
    _wait_for_worker(page)

    stages = [record[1] for record in page.progress_panel.records]
    assert stages.index("load_project_data") < stages.index("equipment_grouping") < stages.index("calculate_kpi")
    assert stages.index("calculate_kpi") < stages.index("diagnostics") < stages.index("finalize")
    assert page.progress_panel.statuses["finalize"] == "completed"
    assert page.progress_panel.bar.value() == 100
    assert page.run_button.isEnabled()
    assert page.tabs.isVisible()
    assert page.cop_chart.data == [("CH-1", page.cop_chart.data[0][1])]
    assert page.delta_chart.data[0][1] is not None
    assert page.kpi_layout.count() > 1
    page.close()


def test_analysis_worker_failure_keeps_progress_and_recovers_button(qapp, tmp_path):
    context = _context(tmp_path)

    def fail(_callback):
        raise RuntimeError("broken analysis input")

    context.run_diagnosis = fail
    page = AnalyticsPage(context); page.show(); page.run_diagnosis(); _wait_for_worker(page)
    assert page.progress_panel.failure[0] == "load_project_data"
    assert page.run_button.isEnabled()
    page.close()


def test_progress_panel_translates_live_and_prevents_duplicate_workers(qapp, tmp_path):
    context = _context(tmp_path)
    calls = []

    def slow(callback):
        calls.append(1)
        callback("load_project_data", "running", None, 0, 0)
        time.sleep(.12)
        callback("finalize", "completed", None, 1, 1)
        return type("Result", (), {"analytics": type("Analytics", (), {"equipment_kpis": []})(), "findings": []})()

    context.run_diagnosis = slow
    page = AnalyticsPage(context); page.show(); page.run_diagnosis(); page.run_diagnosis()
    assert page._analysis_worker is not None
    LanguageManager.instance().set_language("zh_CN")
    assert "加载项目数据" in page.progress_panel.stage_labels["load_project_data"].text()
    LanguageManager.instance().set_language("en_US")
    assert "Loading project data" in page.progress_panel.stage_labels["load_project_data"].text()
    _wait_for_worker(page)
    assert len(calls) == 1
    page.close()


def test_one_equipment_error_is_reported_without_aborting_other_equipment(tmp_path):
    context = _context(tmp_path)
    frame = context.dataframe.copy()
    points = list(context.semantic_result.semantic_results)
    for point in list(points):
        raw = point.raw_name.replace("CH-1", "CH-2")
        column = (point.column or point.raw_name).replace("CH-1", "CH-2")
        frame[column] = frame[point.column or point.raw_name]
        points.append(SemanticResult(raw, point.canonical_label, point_id="2-" + point.point_id, unit=point.unit, column=column))
    semantics = AnalysisResult(points, {})
    service = AnalyticsService()
    original = service._analyze_heat_source

    def fail_second(df, binding, timestamps):
        if binding.equipment.name == "CH-2":
            raise ValueError("bad signal")
        return original(df, binding, timestamps)

    service._analyze_heat_source = fail_second
    result = service.analyze_project(frame, semantics, "project", {"time_column": "time"}, EquipmentService().organize("project", points))
    assert [item.status for item in result.equipment_kpis].count("available") == 1
    assert [item.status for item in result.equipment_kpis].count("skipped") == 1
