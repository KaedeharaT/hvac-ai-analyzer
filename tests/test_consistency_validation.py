import pandas as pd

from building_ai.models import AnalysisResult, SemanticResult
from building_ai.services.analytics_service import AnalyticsService
from building_ai.services.consistency_validation import validate_analysis
from building_ai.services.diagnosis_service import DiagnosisService
from building_ai.services.opportunity_service import OpportunityService


def _inputs():
    frame = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=6, freq="h"),
        "CH-1 Supply": [6.0] * 6, "CH-1 Return": [8.0] * 6,
        "CH-1 Flow": [3.6] * 6, "CH-1 Power": [8.0] * 6,
    })
    points = [
        SemanticResult("CH-1 Supply", "heat_source_supply_temp", point_id="s", unit="°C", column="CH-1 Supply"),
        SemanticResult("CH-1 Return", "heat_source_return_temp", point_id="r", unit="°C", column="CH-1 Return"),
        SemanticResult("CH-1 Flow", "heat_source_flow", point_id="f", unit="m³/h", column="CH-1 Flow"),
        SemanticResult("CH-1 Power", "heat_source_power", point_id="p", unit="kW", column="CH-1 Power"),
    ]
    return frame, AnalysisResult(points, {})


def test_canonical_metric_summary_uses_the_same_valid_mask_as_cop():
    frame, semantics = _inputs()
    analytics = AnalyticsService().analyze_project(frame, semantics, "project", {"time_column": "time"})
    item = analytics.equipment_kpis[0]
    assert item.valid_sample_count == int(item.cop.series.notna().sum())
    assert item.metric_summary["delta_t_c"]["mean"] == item.cop.delta_t_c.where(item.valid_mask).mean()
    assert item.metric_summary["power_kw"]["mean"] == item.cop.input_power_kw.where(item.valid_mask).mean()


def test_consistency_validator_rejects_cross_layer_delta_t_mismatch():
    frame, semantics = _inputs()
    diagnosis = DiagnosisService().diagnose_project(frame, semantics, "project", {"time_column": "time"})
    opportunities = OpportunityService(enable_llm=False).identify(diagnosis.findings)
    assert validate_analysis(diagnosis.analytics, diagnosis.findings, opportunities).passed

    item = diagnosis.analytics.equipment_kpis[0]
    item.metric_summary["delta_t_c"]["mean"] = 99.0
    result = validate_analysis(diagnosis.analytics, diagnosis.findings, opportunities)
    assert not result.passed
    assert any(issue.code == "metric_summary_mismatch" for issue in result.issues)

