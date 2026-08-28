import pandas as pd

from building_ai.services import SemanticService


def test_single_pipeline_returns_unified_result():
    frame = pd.DataFrame({
        "CH-1_LWT": [6.8, 7.0], "冷水回水温度": [12.0, 12.2],
        "mystery": [1, 2],
    })
    result = SemanticService().analyze_dataframe(frame)
    assert result.role_dict["CH-1_LWT"] == "heat_source_supply_temp"
    assert result.role_dict["冷水回水温度"] == "heat_source_return_temp"
    assert result.by_raw_name()["mystery"].status.value == "ABSTAIN"
    assert set(result.role_dict) == set(result.ai_roles) == set(frame.columns)


def test_generic_quantities_do_not_default_to_heat_source_accept():
    frame = pd.DataFrame({"電力": [10, 11], "流量": [2, 3], "Capacity": [100, 100]})
    result = SemanticService().analyze_dataframe(frame)
    for name in frame.columns:
        item = result.by_raw_name()[name]
        assert item.canonical_label == "other"
        assert item.equipment_type is None
        assert item.status.value in {"REVIEW", "ABSTAIN"}


def test_explicit_equipment_tokens_are_mapped():
    frame = pd.DataFrame({"CH-1 Power": [10, 11], "AHU-01 Power": [2, 3]})
    result = SemanticService().analyze_dataframe(frame)
    assert result.role_dict["CH-1 Power"] == "heat_source_power"
    assert result.role_dict["AHU-01 Power"] == "terminal_power"
    assert all(item.status.value == "ACCEPT" for item in result.semantic_results)


def test_capacity_is_other_in_the_current_strict_research_taxonomy():
    result = SemanticService().analyze_dataframe(pd.DataFrame({"CH-1 Capacity kW": [100, 100]}))
    assert result.role_dict["CH-1 Capacity kW"] == "other"
