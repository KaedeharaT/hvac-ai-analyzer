import pandas as pd
from types import SimpleNamespace

from building_ai.models import AnalysisResult, SemanticResult
from building_ai.services.diagnosis_service import DiagnosisService
from building_ai.services.opportunity_service import OpportunityService
from building_ai.core.diagnostics import DiagnosisConfig


def _semantics(with_capacity=False):
    items = [
        SemanticResult("CH-1 Supply °C", "heat_source_supply_temp", point_id="s", unit="°C", column="CH-1 Supply °C"),
        SemanticResult("CH-1 Return °C", "heat_source_return_temp", point_id="r", unit="°C", column="CH-1 Return °C"),
        SemanticResult("CH-1 Flow m3/h", "heat_source_flow", point_id="f", unit="m³/h", column="CH-1 Flow m3/h"),
        SemanticResult("CH-1 Power kW", "heat_source_power", point_id="p", unit="kW", column="CH-1 Power kW"),
    ]
    if with_capacity:
        items.append(SemanticResult("CH-1 Capacity kW", "heat_source_capacity", point_id="c", unit="kW", column="CH-1 Capacity kW"))
    return AnalysisResult(items, {item.raw_name: item.canonical_label for item in items})


def _frame(delta_t=2.0, power=10.0, with_time=True, with_capacity=False):
    values = {
        "CH-1 Supply °C": [6.0] * 6,
        "CH-1 Return °C": [6.0 + delta_t] * 6,
        "CH-1 Flow m3/h": [3.6] * 6,
        "CH-1 Power kW": [power] * 6,
    }
    if with_time:
        values["time"] = pd.date_range("2026-01-05 01:00", periods=6, freq="h")
    if with_capacity:
        values["CH-1 Capacity kW"] = [100.0] * 6
    return pd.DataFrame(values)


def test_diagnosis_detects_low_cop_low_delta_t_and_offhour_operation():
    result = DiagnosisService(config=DiagnosisConfig(workday_start_hour=8, workday_end_hour=19)).diagnose_project(
        _frame(), _semantics(), "project", {"time_column": "time"}
    )
    types = {item.finding_type for item in result.findings}
    assert {"low_heat_source_cop", "low_chilled_water_delta_t", "off_hour_operation"} <= types
    assert all(item.evidence and item.evidence[0].point_ids for item in result.findings)


def test_diagnosis_normal_cop_and_missing_time_are_not_false_positives():
    result = DiagnosisService().diagnose_project(
        _frame(delta_t=6.0, power=3.0, with_time=False), _semantics(), "project", {}
    )
    types = {item.finding_type for item in result.findings}
    assert "low_heat_source_cop" not in types
    assert "low_chilled_water_delta_t" not in types
    assert any("time column" in item for item in result.skipped)


def test_diagnosis_requires_a_project_schedule_for_offhour_and_aggregates_findings():
    result = DiagnosisService().diagnose_project(
        _frame(), _semantics(), "project", {"time_column": "time"}
    )
    assert "off_hour_operation" not in {item.finding_type for item in result.findings}
    assert any("schedule is unavailable" in item for item in result.skipped)

    # One rule result represents the complete event period for this equipment;
    # it must not emit one warning per timestamp.
    keys = [(item.equipment_id, item.finding_type) for item in result.findings]
    assert len(keys) == len(set(keys))
    for item in result.findings:
        assert item.equipment_id and item.finding_id and item.severity
        assert item.evidence and item.affected_period


def test_diagnosis_skips_when_cop_inputs_are_insufficient():
    frame = _frame().drop(columns=["CH-1 Flow m3/h"])
    result = DiagnosisService().diagnose_project(frame, _semantics(), "project", {"time_column": "time"})
    assert not result.findings
    assert any("Missing" in item or "No" in item for item in result.skipped)


def test_diagnosis_detects_low_load_high_power_with_rated_capacity():
    frame = _frame(delta_t=6.0, power=10.0, with_capacity=True)
    frame.loc[:2, "CH-1 Return °C"] = 8.0
    result = DiagnosisService().diagnose_project(
        frame, _semantics(with_capacity=True), "project", {"time_column": "time"}
    )
    assert "low_load_high_power" in {item.finding_type for item in result.findings}


def test_diagnosis_detects_short_cycling_and_low_load_parallel_operation():
    frame = _frame(delta_t=6.0, power=3.0, with_capacity=True)
    for role, unit in (("Supply °C", "°C"), ("Return °C", "°C"), ("Flow m3/h", "m³/h"), ("Power kW", "kW"), ("Capacity kW", "kW")):
        source = f"CH-1 {role}"
        target = f"CH-2 {role}"
        frame[target] = frame[source]
    items = _semantics(with_capacity=True).semantic_results
    second = [
        SemanticResult(item.raw_name.replace("CH-1", "CH-2"), item.canonical_label,
                       point_id=f"2-{item.point_id}", unit=item.unit,
                       column=(item.column or item.raw_name).replace("CH-1", "CH-2"))
        for item in items
    ]
    semantics = AnalysisResult(items + second, {})
    result = DiagnosisService().diagnose_project(frame, semantics, "project", {"time_column": "time"})
    assert "parallel_heat_source_operation" in {item.finding_type for item in result.findings}

    cycling = _frame(delta_t=6.0, power=3.0)
    cycling["CH-1 Power kW"] = [3, 0, 3, 0, 3, 0]
    cycle_result = DiagnosisService().diagnose_project(cycling, _semantics(), "project", {"time_column": "time"})
    assert "frequent_start_stop" in {item.finding_type for item in cycle_result.findings}


def test_opportunity_mapping_survives_local_llm_unavailable():
    diagnosis = DiagnosisService().diagnose_project(_frame(), _semantics(), "project", {"time_column": "time"})
    opportunities = OpportunityService(enable_llm=False).identify(diagnosis.findings)
    assert opportunities
    assert all(item.recommendation for item in opportunities)
    assert all(item.llm_explanation["status"] == "skipped" for item in opportunities)


def test_opportunity_mapping_survives_mocked_local_llm_failure():
    class FailingLLM:
        settings = SimpleNamespace(provider="ollama", model="local-model")

        def chat_json(self, *args, **kwargs):
            raise RuntimeError("Ollama is unavailable")

    diagnosis = DiagnosisService().diagnose_project(_frame(), _semantics(), "project", {"time_column": "time"})
    opportunities = OpportunityService(FailingLLM()).identify(diagnosis.findings)
    assert opportunities
    assert all(item.llm_explanation["status"] == "unavailable" for item in opportunities)
