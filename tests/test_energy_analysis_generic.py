from pathlib import Path

import pandas as pd
import pytest

from building_ai.models import AnalysisResult, SemanticResult
from building_ai.services.energy_analysis_service import EnergyAnalysisService
from building_ai.services.analytics_service import AnalyticsService
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


def test_resampling_uses_quantity_correct_aggregation_and_scopes_summary():
    timestamps = pd.date_range("2025-01-01", periods=30, freq="min")
    frame = pd.DataFrame({
        "timestamp": timestamps,
        "Power": range(30),
        "Interval Energy": [1.0] * 30,
        "Temperature": range(20, 50),
    })
    points = [
        _point("Power", "heat_source_power", "power", "kW", "CH-1"),
        _point("Interval Energy", "heat_source_energy", "energy", "kWh", "METER-1"),
        _point("Temperature", "heat_source_supply_temp", "temperature", "°C", "CH-1"),
    ]
    result = EnergyAnalysisService().analyze(
        frame, AnalysisResult(points), "scope", {"time_column": "timestamp"},
        EquipmentService().organize("scope", points), aggregation="10min",
        period_start=timestamps[0], period_end=timestamps[-1],
    )
    power = result.charts["power_trend"]["series"][0]["data"]
    energy = result.charts["energy_trend"]["series"][0]["data"]
    temperature = result.charts["temperature_trend"]["series"][0]["data"]
    assert [row["value"] for row in power] == [4.5, 14.5, 24.5]  # mean kW
    assert [row["value"] for row in energy] == pytest.approx([10.75, 12.4166666667, 14.0833333333])
    assert [row["value"] for row in temperature] == [24.5, 34.5, 44.5]  # mean °C
    assert result.charts["power_trend"]["aggregation_operation"] == "mean"
    assert result.charts["power_trend"]["peak_operation"] == "raw_max"
    assert result.charts["energy_trend"]["aggregation_operation"] == "sum"
    assert result.charts["temperature_trend"]["aggregation_operation"] == "mean"
    assert result.summary["peak_power_kw"] == 29.0  # peak remains a raw maximum
    assert result.aggregation_rule == "10min"


def test_cumulative_energy_is_differenced_before_custom_period_sum():
    timestamps = pd.date_range("2025-01-01", periods=60, freq="min")
    frame = pd.DataFrame({"timestamp": timestamps, "Meter": range(60)})
    points = [_point("Meter", "heat_source_energy", "energy", "kWh", "METER-1")]
    result = EnergyAnalysisService().analyze(
        frame, AnalysisResult(points), "scope", {"time_column": "timestamp"},
        aggregation="10min", period_start=timestamps[10], period_end=timestamps[29],
    )
    assert result.summary["total_energy_kwh"] == 20.0
    assert [row["value"] for row in result.charts["energy_trend"]["series"][0]["data"]] == [10.0, 10.0]


def test_one_minute_resolution_preserves_native_one_minute_power():
    timestamps = pd.date_range("2025-01-01", periods=4, freq="min")
    frame = pd.DataFrame({"timestamp": timestamps, "Power": [10.0, 20.0, 30.0, 40.0]})
    points = [_point("Power", "heat_source_power", "power", "kW", "CH-1")]
    result = EnergyAnalysisService().analyze(frame, AnalysisResult(points), "native", {"time_column": "timestamp"}, aggregation="1min")
    assert result.aggregation_rule == "1min"
    assert [row["value"] for row in result.charts["power_trend"]["series"][0]["data"]] == [10.0, 20.0, 30.0, 40.0]


def test_cop_resampling_is_mean_and_retains_valid_sample_count():
    timestamps = pd.date_range("2025-01-01", periods=30, freq="min")
    frame = pd.DataFrame({"timestamp": timestamps, "Supply": [7.0] * 30, "Return": [12.0] * 30,
                          "Flow": [10.0] * 30, "Power": [50.0] * 30})
    points = [
        _point("Supply", "heat_source_supply_temp", "temperature", "°C", "CH-1"),
        _point("Return", "heat_source_return_temp", "temperature", "°C", "CH-1"),
        _point("Flow", "heat_source_flow", "flow", "L/s", "CH-1"),
        _point("Power", "heat_source_power", "power", "kW", "CH-1"),
    ]
    semantics = AnalysisResult(points); organization = EquipmentService().organize("cop", points)
    analytics = AnalyticsService().analyze_project(frame, semantics, "cop", {"time_column": "timestamp"}, organization)
    result = EnergyAnalysisService().analyze(frame, semantics, "cop", {"time_column": "timestamp"}, organization, analytics, aggregation="10min")
    rows = result.charts["cop_trend"]["series"][0]["data"]
    assert len(rows) == 3
    assert all(row["valid_sample_count"] == 10 for row in rows)
    assert all(abs(row["value"] - 4.186) < 1e-9 for row in rows)
    assert result.summary["average_cop_valid_sample_count"] == 30


def test_custom_period_comparison_uses_explicit_non_overlapping_bounds():
    timestamps = pd.date_range("2025-01-01", periods=4, freq="h")
    frame = pd.DataFrame({"timestamp": timestamps, "Power": [10.0, 10.0, 20.0, 20.0]})
    points = [_point("Power", "heat_source_power", "power", "kW", "CH-1")]
    result = EnergyAnalysisService().analyze(
        frame, AnalysisResult(points), "comparison", {"time_column": "timestamp"},
        comparison_periods={"a": (timestamps[0], timestamps[1]), "b": (timestamps[2], timestamps[3])},
    )
    comparison = result.charts["period_comparison"]
    assert comparison["basis"] == "custom_periods"
    assert [row["value"] for row in comparison["data"]] == [10.0, 20.0]
    assert [row["valid_sample_count"] for row in comparison["data"]] == [2, 2]


def test_invalid_custom_period_is_rejected():
    timestamps = pd.date_range("2025-01-01", periods=4, freq="h")
    frame = pd.DataFrame({"timestamp": timestamps, "Power": [10.0] * 4})
    points = [_point("Power", "heat_source_power", "power", "kW")]
    try:
        EnergyAnalysisService().analyze(frame, AnalysisResult(points), "invalid", {"time_column": "timestamp"}, period_start=timestamps[3], period_end=timestamps[0])
    except ValueError as exc:
        assert "period_start" in str(exc)
    else:
        raise AssertionError("invalid period must not be accepted")


@pytest.mark.parametrize("aggregation, expected_rule, start, periods, frequency, expected_bins", [
    ("1min", "1min", "2025-01-01", 20, "min", 20),
    ("10min", "10min", "2025-01-01", 20, "min", 2),
    ("1h", "h", "2025-01-01", 12, "10min", 2),
    ("1d", "D", "2025-01-01", 48, "h", 2),
    ("1w", "W-MON", "2025-01-06", 14, "D", 2),
    ("1mo", "MS", "2025-01-01", 59, "D", 2),
    ("1y", "YS", "2024-01-01", 731, "D", 2),
])
def test_every_supported_resolution_changes_the_real_data_bins(aggregation, expected_rule, start, periods, frequency, expected_bins):
    timestamps = pd.date_range(start, periods=periods, freq=frequency)
    frame = pd.DataFrame({"timestamp": timestamps, "Power": [60.0] * len(timestamps)})
    points = [_point("Power", "heat_source_power", "power", "kW", "CH-1")]
    result = EnergyAnalysisService().analyze(frame, AnalysisResult(points), "bins", {"time_column": "timestamp"}, aggregation=aggregation)
    assert result.aggregation_rule == expected_rule
    assert len(result.charts["power_trend"]["series"][0]["data"]) == expected_bins
    native_minutes = EnergyAnalysisService._interval_minutes(pd.Series(timestamps))
    assert result.summary["total_energy_kwh"] == pytest.approx(60.0 * periods * native_minutes / 60.0)


def test_native_resolution_protection_rejects_upsampling():
    timestamps = pd.date_range("2025-01-01", periods=12, freq="10min")
    frame = pd.DataFrame({"timestamp": timestamps, "Power": [10.0] * len(timestamps)})
    points = [_point("Power", "heat_source_power", "power", "kW", "CH-1")]
    with pytest.raises(ValueError, match="upsampling is not permitted"):
        EnergyAnalysisService().analyze(frame, AnalysisResult(points), "native", {"time_column": "timestamp"}, aggregation="1min")


def test_coarse_resolution_does_not_create_misleading_time_of_day_charts():
    timestamps = pd.date_range("2025-01-01", periods=3 * 24, freq="h")
    frame = pd.DataFrame({"timestamp": timestamps, "Power": [10.0] * len(timestamps)})
    points = [_point("Power", "heat_source_power", "power", "kW", "CH-1")]
    result = EnergyAnalysisService().analyze(frame, AnalysisResult(points), "coarse", {"time_column": "timestamp"}, aggregation="1d")
    assert "daily_load_profile" not in result.charts
    assert "load_heatmap" not in result.charts
    assert result.capability_details["daily_profile"]["reason"] == "requires_subdaily_resolution"
    assert result.capability_details["heatmap"]["reason"] == "requires_subdaily_resolution"


def test_weather_scatter_uses_selected_resolution_for_both_axes():
    timestamps = pd.date_range("2025-01-01", periods=18, freq="5min")
    frame = pd.DataFrame({"timestamp": timestamps, "Outdoor": range(18), "Power": range(10, 28)})
    points = [
        _point("Outdoor", "other", "temperature", "°C"),
        _point("Power", "heat_source_power", "power", "kW", "CH-1"),
    ]
    result = EnergyAnalysisService().analyze(frame, AnalysisResult(points), "weather", {"time_column": "timestamp"}, aggregation="10min")
    scatter = result.charts["weather_correlation"]
    assert scatter["sample_count"] == 9
    assert scatter["aggregation"] == "10min"
    assert [row["x"] for row in scatter["data"]] == [0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5, 16.5]


def test_week_month_and_year_buckets_use_stable_calendar_boundaries():
    service = EnergyAnalysisService()
    weekly = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-08-23", "2026-08-24"]))  # Sunday, Monday
    assert [row["time"][:10] for row in service._aggregate(weekly, "W-MON")] == ["2026-08-17", "2026-08-24"]
    monthly = pd.Series([1.0, 2.0], index=pd.to_datetime(["2025-12-31", "2026-01-01"]))
    assert [row["time"][:10] for row in service._aggregate(monthly, "MS")] == ["2025-12-01", "2026-01-01"]
    yearly = pd.Series([1.0, 2.0], index=pd.to_datetime(["2025-12-31", "2026-01-01"]))
    assert [row["time"][:10] for row in service._aggregate(yearly, "YS")] == ["2025-01-01", "2026-01-01"]
