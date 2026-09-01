from building_ai.config import Settings
from building_ai.e2e_evaluation import E2E_DATASET, E2ERunner


def test_e2e_dataset_is_separate_and_covers_real_agent_paths():
    assert len(E2E_DATASET) >= 50
    categories = {case.category for case in E2E_DATASET}
    assert {'general_chat','general_hvac_knowledge','project_summary','energy_analysis','equipment_kpi','diagnosis','recommendation','memory','rag','missing_data','nonexistent_equipment','abstention','cross_project_isolation','prompt_injection','tool_failure','ambiguous','multi_step'} <= categories
    assert any(case.setup_turns for case in E2E_DATASET)
    assert any(case.expected_source for case in E2E_DATASET)
    assert any(case.mode == 'tool_failure' for case in E2E_DATASET)


def test_e2e_metrics_do_not_mislabel_structural_checks_as_hallucination_rate():
    """A real LLM run needs a separately frozen factuality protocol."""
    result = E2ERunner(object(), Settings(provider="not_configured"), dataset=()).run()
    assert "Hallucination Rate" not in result["metrics"]
    assert result["metrics"]["Factual Exact-Match Coverage"] == 0.0
    assert result["metrics"]["Factual Exact-Match Accuracy"] == "N/A"
