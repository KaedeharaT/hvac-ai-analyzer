from building_ai.evaluation import EVAL_DATASET, run_deterministic_evaluation


def test_deterministic_agent_evaluation_has_sixty_meaningful_runtime_cases(tmp_path):
    output=tmp_path/'evaluation.json'; result=run_deterministic_evaluation(output)
    assert len(EVAL_DATASET) >= 60
    assert result['case_count'] == len(EVAL_DATASET)
    assert result['failed_cases'] == []
    assert {'general_chat','general_hvac','project_summary','energy_analysis','equipment_analysis','diagnosis','recommendation','memory','rag','missing_data','nonexistent_equipment','prompt_injection','tool_failure'} <= {case.category for case in EVAL_DATASET}
    assert {'Total Cases','Routing Accuracy','Tool Selection Accuracy','Task Success Rate','Grounded Answer Rate','Factual Exact-Match Coverage','Factual Exact-Match Accuracy','Abstention Accuracy','Tool Failure Rate','Average Tool Calls','Average Agent Latency','Average LLM Latency','RAG Retrieval Hit Rate'} == set(result['metrics'])
    assert result['metrics']['Total Cases'] >= 60
    assert result['metrics']['Routing Accuracy'] == 1.0
    assert result['metrics']['Factual Exact-Match Coverage'] > 0.0
    assert result['metrics']['Factual Exact-Match Accuracy'] == 1.0
    assert output.exists()
    assert (tmp_path/'latest.md').exists()
