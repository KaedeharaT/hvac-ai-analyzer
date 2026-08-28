from pathlib import Path

import pandas as pd

from building_ai.config import Settings
from building_ai.services import BuildingAnalysisPipeline, SemanticService


def test_v3_non_target_gate_rejects_control_and_capacity():
    frame = pd.DataFrame({
        "CH-1 Capacity kW": [100.0, 100.0],
        "AHU-1 Supply Air Temp Setpoint": [18.0, 18.0],
    })
    result = SemanticService().analyze_dataframe(frame)
    assert {item.canonical_label for item in result.semantic_results} == {"other"}
    assert all("non_target" in item.reason for item in result.semantic_results)


def test_enhanced_mode_without_llm_falls_back_without_crashing():
    settings = Settings(provider="not_configured")
    result = SemanticService(settings).analyze_dataframe(
        pd.DataFrame({"AHU-01 Supply Air Temp": [17.0, 18.0]}), backend="enhanced"
    )
    item = result.semantic_results[0]
    assert item.canonical_label == "terminal_supply_air_temp"
    assert item.model_provider == "offline"


def test_research_fixture_runs_through_product_pipeline():
    fixture = Path("paper_research/tests/fixtures/bems_v2_smoke.csv")
    frame = pd.read_csv(fixture)
    run = BuildingAnalysisPipeline().run(frame, project_id="research-smoke")
    assert len(run.semantics.semantic_results) == len(frame.columns) - 1
    assert run.semantics.warnings == ["timestamp_column_excluded:timestamp"]
    assert run.semantics.by_raw_name()["AHU supply air temp"].canonical_label == "terminal_supply_air_temp"
    assert run.analytics is not None and run.diagnosis is not None
