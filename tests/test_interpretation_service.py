from building_ai.models import AnalysisEvidence, DiagnosticFinding
from building_ai.services.interpretation_service import InterpretationService


def test_low_delta_t_interpretation_is_actionable_and_site_safe():
    finding = DiagnosticFinding("f-1", "equipment-1", "low_chilled_water_delta_t", "warning", "", "", [AnalysisEvidence([], "equipment-1", "2025-01-01", "2025-01-02", "delta_t", {"mean": 4.0, "low_delta_t_ratio": .2}, "°C")], .9, valid_sample_count=100, occurrence_count=20)
    result = InterpretationService().interpret(finding, language="zh_CN")
    assert result.problem == "冷冻水利用效率偏低"
    assert result.priority == "P1"
    assert result.risk_level == 2
    assert any("小幅" in action for action in result.actions)
    assert "诊断代码" in result.technical_details
