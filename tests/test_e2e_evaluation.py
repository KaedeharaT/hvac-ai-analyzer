from building_ai.e2e_evaluation import E2E_DATASET


def test_e2e_dataset_is_separate_and_covers_real_agent_paths():
    assert len(E2E_DATASET) >= 50
    categories = {case.category for case in E2E_DATASET}
    assert {'general_chat','general_hvac_knowledge','project_summary','energy_analysis','equipment_kpi','diagnosis','recommendation','memory','rag','missing_data','nonexistent_equipment','abstention','cross_project_isolation','prompt_injection','tool_failure','ambiguous','multi_step'} <= categories
    assert any(case.setup_turns for case in E2E_DATASET)
    assert any(case.expected_source for case in E2E_DATASET)
    assert any(case.mode == 'tool_failure' for case in E2E_DATASET)
