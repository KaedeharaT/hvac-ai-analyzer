import pandas as pd

from building_ai.core.cop_engine import calculate_heat_source_cop


def _frame(power=10):
    return pd.DataFrame({"s": [6.0, 6.0, 6.0], "r": [12.0, 12.0, 12.0], "f": [1.0, 1.0, 1.0], "p": [power] * 3})


def test_cop_supports_lps_and_kw():
    result = calculate_heat_source_cop(_frame(), "s", "r", "f", "p", "L/s", "kW", "°C")
    assert result.available
    assert round(result.summary["mean"], 3) == 2.512
    assert result.valid_count == 3


def test_cop_converts_lpm_m3h_and_w():
    frame = _frame(power=10000)
    frame["f"] = 60.0
    lpm = calculate_heat_source_cop(frame, "s", "r", "f", "p", "L/min", "W", "°C")
    frame["f"] = 3.6
    m3h = calculate_heat_source_cop(frame, "s", "r", "f", "p", "m³/h", "W", "℃")
    assert lpm.available and m3h.available
    assert round(lpm.summary["mean"], 3) == round(m3h.summary["mean"], 3) == 2.512


def test_cop_rejects_unknown_units_missing_fields_and_reversed_delta_t():
    unknown_flow = calculate_heat_source_cop(_frame(), "s", "r", "f", "p", None, "kW", "°C")
    unknown_power = calculate_heat_source_cop(_frame(), "s", "r", "f", "p", "L/s", None, "°C")
    missing = calculate_heat_source_cop(_frame(), "s", "r", "missing", "p", "L/s", "kW", "°C")
    reversed_frame = _frame(); reversed_frame["r"] = 4.0
    reversed_result = calculate_heat_source_cop(reversed_frame, "s", "r", "f", "p", "L/s", "kW", "°C")
    assert not unknown_flow.available and "flow unit" in unknown_flow.reason
    assert not unknown_power.available and "power unit" in unknown_power.reason
    assert not missing.available and "Missing" in missing.reason
    assert not reversed_result.available
    assert any("non_positive" in item for item in reversed_result.warnings)
