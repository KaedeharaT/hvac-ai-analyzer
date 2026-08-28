from pathlib import Path

import pandas as pd

from building_ai.models import AnalysisResult, SemanticResult
from building_ai.services.energy_analysis_service import EnergyAnalysisService
from building_ai.services.equipment_service import EquipmentService


def _point(name, label, quantity, unit, equipment=None):
    return SemanticResult(name, label, point_id=name, column=name, physical_quantity=quantity, unit=unit, equipment_id=equipment)


def test_power_energy_are_not_confused_and_interval_is_integrated():
    frame = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=4, freq="30min"), "CH-01 Power kW": [10., 10., 10., 10.], "CH-02 Energy kWh": [100., 105., 110., 115.]})
    points = [_point("CH-01 Power kW", "heat_source_power", "power", "kW", "CH-01"), _point("CH-02 Energy kWh", "other", "energy", "kWh", "CH-02")]
    result = EnergyAnalysisService().analyze(frame, AnalysisResult(points), "project-a", {"time_column": "timestamp"}, EquipmentService().organize("project-a", points))
    # Four 30-minute power samples are 20 kWh; cumulative meter increments are 15 kWh.
    assert result.summary["total_energy_kwh"] == 35.0
    assert result.capabilities["energy_timeseries"]
    assert result.capabilities["power_timeseries"]


def test_unknown_units_are_not_calculated_but_temperature_remains_visible():
    frame = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=4, freq="h"), "Room temperature": [20., 21., 22., 21.], "Plant power": [1., 2., 3., 4.]})
    points = [_point("Room temperature", "other", "temperature", None), _point("Plant power", "other", "power", None)]
    result = EnergyAnalysisService().analyze(frame, AnalysisResult(points), "project-b", {"time_column": "timestamp"})
    assert result.capabilities["temperature_timeseries"]
    assert not result.capabilities["power_timeseries"]
    assert "unknown_or_unsupported_power_unit" in result.warnings
    assert result.capability_details["power_trend"]["available"] is False
    assert result.capability_details["power_trend"]["reason"] == "requires_supported_power_timeseries"


def test_capability_model_and_period_comparison_are_real_result_outputs():
    frame = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=8 * 24, freq="h"), "Plant Power kW": [10.0] * (8 * 24)})
    points = [_point("Plant Power kW", "other", "power", "kW")]
    result = EnergyAnalysisService().analyze(frame, AnalysisResult(points), "project-a", {"time_column": "timestamp"})
    assert result.capabilities["energy_consumption"] is True
    assert result.capability_details["energy_consumption"]["reason"] == "available"
    assert result.charts["period_comparison"]["data"]


def test_equipment_ids_are_project_scoped_and_multilingual():
    assert EquipmentService().organize("project-a", [_point("冷凍機No.1 往水温度 °C", "heat_source_supply_temp", "temperature", "°C", "冷凍機-1")]).equipment[0].project_id == "project-a"
    # UUID identity includes project_id even when the normalized display ID is the same.
    points = [_point("CH-01 Supply °C", "heat_source_supply_temp", "temperature", "°C", "CH-01")]
    assert EquipmentService().organize("project-a", points).equipment[0].equipment_id != EquipmentService().organize("project-b", points).equipment[0].equipment_id


def test_no_project_specific_runtime_constants():
    runtime = Path(__file__).parents[1] / "building_ai"
    forbidden = ("AHP-3-1", "AHP-3-2", "AHP-3-3", "AHP-3-4", "Project 7")
    text = "\n".join(path.read_text(encoding="utf-8") for path in runtime.rglob("*.py"))
    assert not any(token in text for token in forbidden)
