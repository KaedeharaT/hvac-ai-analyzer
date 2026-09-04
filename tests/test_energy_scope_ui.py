import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest
from PyQt5.QtCore import QDateTime
from PyQt5.QtWidgets import QApplication

from building_ai.models import AnalysisResult, SemanticResult
from building_ai.services.energy_analysis_service import EnergyAnalysisService
from building_ai.services.equipment_service import EquipmentService
from building_ai.ui.pages.energy_analysis_page import EnergyAnalysisPage, TimeSeriesChart


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


def _minute_context():
    timestamps = pd.date_range("2025-02-01", periods=2 * 24 * 60, freq="min")
    frame = pd.DataFrame({"timestamp": timestamps, "CH-1 Power": [10.0] * len(timestamps)})
    point = SemanticResult("CH-1 Power", "heat_source_power", point_id="power-minute", column="CH-1 Power",
                           physical_quantity="power", unit="kW", equipment_id="CH-1")
    semantics = AnalysisResult([point]); organization = EquipmentService().organize("p-minute", [point])
    return SimpleNamespace(
        dataframe=frame, semantic_result=semantics, current_project=SimpleNamespace(project_id="p-minute"),
        import_metadata={"time_column": "timestamp"}, equipment_organization=organization,
        diagnosis_result=None, energy_analysis=EnergyAnalysisService(), selected_period="all",
        selected_equipment_id=None, energy_analysis_scope={}, energy_analysis_view_result=None,
    )


def test_energy_page_has_one_scope_controller_and_publishes_scoped_result():
    _app(); context = _context(); page = EnergyAnalysisPage(context)
    assert [page.aggregation.itemData(index) for index in range(page.aggregation.count())] == [
        "1min", "10min", "1h", "1d", "1w", "1mo", "1y"
    ]
    assert not page.aggregation.model().item(page.aggregation.findData("1min")).isEnabled()
    assert not page.aggregation.model().item(page.aggregation.findData("10min")).isEnabled()
    page.range.setCurrentIndex(page.range.findData("custom"))
    page.start_time.setDateTime(QDateTime(pd.Timestamp("2025-02-01 12:00").to_pydatetime()))
    page.end_time.setDateTime(QDateTime(pd.Timestamp("2025-02-02 11:00").to_pydatetime()))
    page.aggregation.setCurrentIndex(page.aggregation.findData("1h"))
    page.refresh()
    assert context.energy_analysis_scope["start"].startswith("2025-02-01T12:00")
    assert context.energy_analysis_scope["end"].startswith("2025-02-02T11:00")
    assert context.energy_analysis_scope["aggregation_rule"] == "h"
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


@pytest.mark.parametrize("resolution, value, start, end, expected", [
    ("1min", "2026-09-03 09:15", "2026-09-03", "2026-09-03 12:00", "09:15"),
    ("10min", "2026-09-04 00:10", "2026-09-03 23:30", "2026-09-04 01:00", "09-04 00:10"),
    ("1h", "2026-09-03 06:00", "2026-09-02", "2026-09-04", "09-03 06:00"),
    ("1d", "2026-01-01", "2025-12-30", "2026-01-02", "2026-01-01"),
    ("1w", "2026-08-17", "2026-08-01", "2026-09-01", "2026-W34"),
    ("1mo", "2026-08-01", "2026-01-01", "2026-12-01", "2026-08"),
    ("1y", "2026-01-01", "2024-01-01", "2026-12-31", "2026"),
])
def test_time_axis_formatter_tracks_resolution(resolution, value, start, end, expected):
    assert TimeSeriesChart.format_time_label(value, resolution, start, end) == expected


def test_dynamic_tick_density_stays_readable():
    indexes = TimeSeriesChart.tick_indexes(10_000, 1000)
    assert 5 <= len(indexes) <= 10
    assert indexes[0] == 0 and indexes[-1] == 9999


@pytest.mark.parametrize("resolution", ["1min", "10min", "1h", "1d", "1w", "1mo", "1y"])
def test_gui_can_apply_every_resolution_without_changing_scope_contract(resolution):
    _app(); context = _minute_context(); page = EnergyAnalysisPage(context)
    page.aggregation.setCurrentIndex(page.aggregation.findData(resolution)); page.refresh()
    assert context.energy_analysis_view_result.aggregation == resolution
    assert context.energy_analysis_scope["aggregation"] == resolution


def test_energy_page_renders_every_resolution_at_supported_window_sizes():
    app = _app(); context = _minute_context(); page = EnergyAnalysisPage(context)
    page.show()
    for resolution in ("1min", "10min", "1h", "1d", "1w", "1mo", "1y"):
        page.aggregation.setCurrentIndex(page.aggregation.findData(resolution))
        page.refresh()
        for width, height in ((1280, 720), (1440, 900), (1920, 1080)):
            page.resize(width, height)
            app.processEvents()
            assert not page.grab().isNull()
    page.close()
