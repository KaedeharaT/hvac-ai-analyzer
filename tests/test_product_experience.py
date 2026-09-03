from __future__ import annotations

import pandas as pd
import pytest
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication

from building_ai.config import Settings
from building_ai.models import AnalysisResult, SemanticResult
from building_ai.services.product_experience import AnalysisReportService, analysis_readiness, passed_checks
from building_ai.ui.context import ApplicationContext
from building_ai.ui.main_window import MainWindow
from building_ai.vision import DrawingDetection


_APP = None


@pytest.fixture(scope="module")
def qapp():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def product_context(tmp_path):
    context = ApplicationContext(Settings(provider="not_configured", data_dir=tmp_path / "data"))
    context.current_project = context.projects.create("Synthetic product QA")
    context.dataframe = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=8, freq="h"),
        "CH-1 Supply °C": [6.0] * 8,
        "CH-1 Return °C": [13.0] * 8,
        "CH-1 Flow m3/h": [3.6] * 8,
        "CH-1 Power kW": [2.0] * 8,
    })
    points = [
        SemanticResult("CH-1 Supply °C", "heat_source_supply_temp", point_id="s", unit="°C", column="CH-1 Supply °C", equipment_id="CH-1"),
        SemanticResult("CH-1 Return °C", "heat_source_return_temp", point_id="r", unit="°C", column="CH-1 Return °C", equipment_id="CH-1"),
        SemanticResult("CH-1 Flow m3/h", "heat_source_flow", point_id="f", unit="m³/h", column="CH-1 Flow m3/h", equipment_id="CH-1"),
        SemanticResult("CH-1 Power kW", "heat_source_power", point_id="p", unit="kW", column="CH-1 Power kW", equipment_id="CH-1"),
    ]
    context.semantic_result = AnalysisResult(points, {item.raw_name: item.canonical_label for item in points})
    context.import_metadata = {"time_column": "time"}
    context.equipment_organization = context.equipment_service.organize(context.current_project.project_id, points)
    context.equipment = context.equipment_organization.equipment
    context.run_diagnosis()
    return context


def test_passed_checks_only_include_evaluated_non_triggered_rules(tmp_path):
    context = product_context(tmp_path)
    checks = passed_checks(context.diagnosis_result)
    assert {item.rule_id for item in checks} == {"low_heat_source_cop", "low_chilled_water_delta_t"}
    assert all("healthy" not in item.title.lower() for item in checks)


def test_capability_readiness_and_report_preserve_evidence_boundaries(tmp_path):
    context = product_context(tmp_path)
    readiness = analysis_readiness(context.energy_analysis_result, context.diagnosis_result)
    assert any(item.capability == "cop" and item.status == "ready" for item in readiness)
    report = AnalysisReportService.markdown(context.current_project, context.energy_analysis_result, context.diagnosis_result, context.opportunities)
    assert "deterministic BuildingAI services" in report
    assert "$" not in report and "supported measured impact calculation" in report
    assert "Project evidence" in report
    html = AnalysisReportService.html(context.current_project, context.energy_analysis_result, context.diagnosis_result, context.opportunities)
    assert html.startswith("<!doctype html>") and "BuildingAI Analysis Report" in html


def test_global_context_navigation_equipment_detail_and_ai_focus(qapp, tmp_path):
    context = product_context(tmp_path)
    window = MainWindow(context)
    equipment = context.equipment[0]
    window.navigate_to("equipment", {"equipment_id": equipment.name})
    page = window.page_by_key["equipment"]
    assert page.detail_title.text() == equipment.name
    assert "COP" in page.detail_views["equipment_performance"].toPlainText()
    window.navigate_to("agent", {"equipment_id": equipment.name, "prompt": "What should I inspect first?"})
    agent = window.page_by_key["agent"]
    assert agent._focus == equipment.name
    assert "inspect" in agent.input.toPlainText()
    window.close()


def test_diagnostics_workbench_exposes_passed_checks_and_filters(qapp, tmp_path):
    context = product_context(tmp_path)
    window = MainWindow(context)
    page = window.page_by_key["analysis"]
    page.refresh()
    assert page.tabs.count() == 5
    assert page.passed_layout.count() > 1
    assert page.finding_table.columnCount() == 5
    window.close()


def test_finding_detail_uses_measured_supporting_trend(qapp, tmp_path):
    context = product_context(tmp_path)
    context.dataframe["CH-1 Return °C"] = [8.1] * 8
    context.dataframe["CH-1 Flow m3/h"] = [18.0] * 8
    context.dataframe["CH-1 Power kW"] = [42.0] * 8
    context.run_diagnosis()
    assert context.diagnosis_result.findings
    window = MainWindow(context); page = window.page_by_key["analysis"]
    page.tabs.setCurrentIndex(2); page.finding_table.selectRow(0); page._show_finding_detail()
    assert page.finding_trend.payload.get("series")
    assert "Measured project trend" in page.finding_detail.toPlainText()
    window.close()


def test_english_interpretations_do_not_leak_chinese_actions(tmp_path):
    context = product_context(tmp_path)
    assert context.user_interpretations
    for item in context.user_interpretations:
        assert all(not any("\u4e00" <= char <= "\u9fff" for char in action) for action in item.actions)


def test_global_ai_entry_preserves_finding_context(qapp, tmp_path):
    context = product_context(tmp_path)
    window = MainWindow(context)
    finding = context.diagnosis_result.findings[0]
    context.selected_finding_id = finding.finding_id
    context.selected_equipment_id = context.equipment[0].name
    window.context_bar.ask_ai.emit()
    assert window.stack.currentWidget() is window.page_by_key["agent"]
    assert context.selected_finding_id == finding.finding_id
    assert window.page_by_key["agent"]._focus == context.equipment[0].name
    window.close()


def test_confirmed_drawing_equipment_cross_navigation(qapp, tmp_path):
    context = product_context(tmp_path)
    image_path = tmp_path / "synthetic-layout.png"
    image = QImage(120, 80, QImage.Format_RGB32); image.fill(0xFFFFFFFF); assert image.save(str(image_path), "PNG")
    drawing = context.drawings.import_drawing(context.current_project.project_id, image_path)
    detection = DrawingDetection("aircon", .9, 10, 10, 60, 50, 120, 80)
    row = context.drawings.save_detections(context.current_project.project_id, drawing["drawing_id"], "fake", [detection])[0]
    context.drawings.review_detection(context.current_project.project_id, row["detection_id"], "confirmed", "aircon")
    context.drawings.map_equipment(context.current_project.project_id, row["detection_id"], context.equipment[0].equipment_id)
    window = MainWindow(context); window.navigate_to("drawing_intelligence")
    page = window.page_by_key["drawing_intelligence"]; page.list.setCurrentRow(0); page.open_selected_equipment()
    assert window.stack.currentWidget() is window.page_by_key["equipment"]
    assert window.page_by_key["equipment"].detail_title.text() == context.equipment[0].name
    window.close()
