import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
from PyQt5.QtCore import QDateTime
from PyQt5.QtWidgets import QApplication

from building_ai.models import AnalysisResult, SemanticResult
from building_ai.services.energy_analysis_service import EnergyAnalysisService
from building_ai.services.equipment_service import EquipmentService
from building_ai.ui.pages.energy_analysis_page import EnergyAnalysisPage


_APP = None


def _app():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _context():
    timestamps = pd.date_range("2025-02-01", periods=48, freq="h")
    frame = pd.DataFrame({"timestamp": timestamps, "CH-1 Power": [10.0] * 24 + [20.0] * 24})
    point = SemanticResult("CH-1 Power", "heat_source_power", point_id="power", column="CH-1 Power",
                           physical_quantity="power", unit="kW", equipment_id="CH-1")
    semantics = AnalysisResult([point]); organization = EquipmentService().organize("p1", [point])
    return SimpleNamespace(
        dataframe=frame, semantic_result=semantics, current_project=SimpleNamespace(project_id="p1"),
        import_metadata={"time_column": "timestamp"}, equipment_organization=organization,
        diagnosis_result=None, energy_analysis=EnergyAnalysisService(), selected_period="all",
        selected_equipment_id=None, energy_analysis_scope={}, energy_analysis_view_result=None,
    )


def test_energy_page_has_one_scope_controller_and_publishes_scoped_result():
    _app(); context = _context(); page = EnergyAnalysisPage(context)
    assert [page.aggregation.itemData(index) for index in range(page.aggregation.count())] == [
        "auto", "raw", "5min", "15min", "30min", "hour", "day", "week", "month"
    ]
    page.range.setCurrentIndex(page.range.findData("custom"))
    page.start_time.setDateTime(QDateTime(pd.Timestamp("2025-02-01 12:00").to_pydatetime()))
    page.end_time.setDateTime(QDateTime(pd.Timestamp("2025-02-02 11:00").to_pydatetime()))
    page.aggregation.setCurrentIndex(page.aggregation.findData("15min"))
    page.refresh()
    assert context.energy_analysis_scope["start"].startswith("2025-02-01T12:00")
    assert context.energy_analysis_scope["end"].startswith("2025-02-02T11:00")
    assert context.energy_analysis_scope["aggregation_rule"] == "15min"
    assert context.energy_analysis_view_result.summary["peak_power_kw"] == 20.0


def test_energy_page_custom_comparison_reaches_service_payload():
    _app(); context = _context(); page = EnergyAnalysisPage(context)
    page.compare_enabled.setChecked(True)
    page.period_a_start.setDateTime(QDateTime(pd.Timestamp("2025-02-01 00:00").to_pydatetime()))
    page.period_a_end.setDateTime(QDateTime(pd.Timestamp("2025-02-01 23:00").to_pydatetime()))
    page.period_b_start.setDateTime(QDateTime(pd.Timestamp("2025-02-02 00:00").to_pydatetime()))
    page.period_b_end.setDateTime(QDateTime(pd.Timestamp("2025-02-02 23:00").to_pydatetime()))
    page.refresh()
    payload = context.energy_analysis_view_result.charts["period_comparison"]
    assert payload["basis"] == "custom_periods"
    assert [item["value"] for item in payload["data"]] == [10.0, 20.0]

