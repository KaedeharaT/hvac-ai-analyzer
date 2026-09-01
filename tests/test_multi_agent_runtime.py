from types import SimpleNamespace

import pytest

from building_ai.agent_runtime import AgentRuntime
from building_ai.agent_runtime_factory import create_agent_runtime
from building_ai.knowledge import KnowledgeService
from building_ai.multi_agent_runtime import (
    DATA_TOOLS, DRAWING_TOOLS, HVAC_TOOLS, KNOWLEDGE_TOOLS, REVIEWER_TOOLS,
    AgentResult, AgentTask, KnowledgeEvidencePacket, MultiAgentRuntime,
    ReviewerAgent, RoleToolExecutor,
)
from building_ai.observability import TraceStore
from building_ai.storage import Database


class Tools:
    def __init__(self, missing=False): self.calls = []; self.missing = missing
    def call(self, name, **kwargs):
        self.calls.append(name)
        if name == "get_equipment_summary": data = [{"equipment": "AHP-3-3"}, {"equipment": "AHP-3-1"}]
        elif name == "get_equipment_kpis": data = {"available": not self.missing, "reason": "power readings absent" if self.missing else None, "equipment_kpis": [{"equipment": "AHP-3-3", "metrics": {"cop": {"mean": 2.6}}}, {"equipment": "AHP-3-1", "metrics": {"cop": {"mean": 3.4}}}]}
        elif name == "get_diagnostic_findings": data = [{"equipment_name": "AHP-3-3", "finding_type": "low_chilled_water_delta_t"}]
        elif name == "get_energy_timeseries": data = {"available": True, "chart": "energy_trend"}
        elif name == "get_energy_opportunities": data = {"available": True, "opportunities": [{"equipment_name": "AHP-3-3", "recommendation": "inspect bypass"}]}
        elif name == "get_equipment_drawing_location": data = {"equipment_id": kwargs.get("equipment_id"), "reliable": True, "locations": [{"file_name": "synthetic-plan.png", "page_number": 1, "reviewed_class": "aircon", "review_status": "confirmed"}, {"file_name": "synthetic-plan.png", "page_number": 1, "class_name": "window", "review_status": "predicted"}]}
        else: data = {"available": True, "tool": name, **kwargs}
        return SimpleNamespace(ok=True, data=data, error=None)


class Controller:
    def _equipment_name(self, query): return "AHP-3-3" if "ahp-3-3" in query.casefold() else None
    def answer(self, query, **kwargs): return "General bounded guidance."


def runtime(tmp_path, *, missing=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    database = Database(tmp_path / "multi.sqlite3"); tools = Tools(missing=missing)
    context = SimpleNamespace(database=database, agent=SimpleNamespace(tools=tools), agent_controller=Controller(), ensure_project_loaded=lambda _: None, settings=SimpleNamespace(agent_mode="multi"))
    KnowledgeService(database).ingest("HVAC Guide", "https://example.invalid/guide", "Inspect bypass valves and pump differential-pressure reset for low delta T.", "guidance", {"source_id": "hvac-guide", "organization": "Test"})
    return MultiAgentRuntime(context), tools, database, context


def roles(trace): return [item["agent_role"] for item in trace["multi_agent"]["results"]]


def test_coordinator_routes_simple_equipment_question_to_data_agent_only(tmp_path):
    agent, tools, database, _ = runtime(tmp_path)
    result = agent.run("What equipment is in the current project?", "project-a", "case-1")
    trace = TraceStore(database).get(result.trace_id)
    assert roles(trace) == ["coordinator", "data_analyst"]
    assert result.tools_used == ["get_equipment_summary"] and "AHP-3-3" in result.answer
    assert "search_knowledge" not in tools.calls


def test_performance_and_recommendation_use_distinct_specialists(tmp_path):
    agent, _, database, _ = runtime(tmp_path)
    performance = agent.run("Which equipment is worst and why?", "project-a", "case-2")
    recommendation = agent.run("How should AHP-3-3 low delta T be improved?", "project-a", "case-3")
    first, second = TraceStore(database).get(performance.trace_id), TraceStore(database).get(recommendation.trace_id)
    assert {"coordinator", "data_analyst", "hvac_expert", "reviewer"} <= set(roles(first))
    assert "knowledge" not in roles(first)
    assert {"coordinator", "data_analyst", "hvac_expert", "knowledge", "reviewer"} <= set(roles(second))
    assert "General reference:" in recommendation.answer


def test_drawing_agent_uses_confirmed_only_and_parent_child_traces(tmp_path):
    agent, _, database, _ = runtime(tmp_path)
    result = agent.run("Where is AHP-3-3 on the drawing?", "project-a", "case-4")
    trace = TraceStore(database).get(result.trace_id)
    drawing = next(item for item in trace["multi_agent"]["results"] if item["agent_role"] == "drawing")
    child = TraceStore(database).get(drawing["trace_id"])
    assert roles(trace) == ["coordinator", "drawing", "reviewer"]
    assert "synthetic-plan.png" in result.answer and not result.abstained
    assert child["parent_trace_id"] == result.trace_id
    assert drawing["packet"]["confirmed_locations"] == [drawing["packet"]["confirmed_locations"][0]]


def test_memory_and_missing_data_abstention(tmp_path):
    agent, _, database, _ = runtime(tmp_path)
    agent.run("How is AHP-3-3 performing?", "project-a", "case-5")
    followup = agent.run("Then what?", "project-a", "case-5")
    assert TraceStore(database).get(followup.trace_id)["memory_used"]
    missing, _, _, _ = runtime(tmp_path / "missing", missing=True)
    unavailable = missing.run("What is the COP of AHP-3-3?", "project-a", "case-6")
    assert unavailable.abstained and "reliable project-specific" in unavailable.answer


def test_reviewer_requests_one_bounded_replan_for_incomplete_engineering_evidence(tmp_path):
    agent, _, database, _ = runtime(tmp_path, missing=True)
    result = agent.run("What is the COP of AHP-3-3?", "project-a", "case-replan")
    trace = TraceStore(database).get(result.trace_id)
    assert trace["multi_agent"]["replan_rounds"] == 1
    assert [item["agent_role"] for item in trace["multi_agent"]["results"]].count("data_analyst") == 2


def test_reviewer_rejects_rag_only_project_diagnosis_and_role_allowlists(tmp_path):
    agent, _, _, context = runtime(tmp_path)
    reviewer = ReviewerAgent(context)
    task = AgentTask(task_id="review", agent_role="reviewer", goal="review", project_id="p", required_output="review")
    knowledge = AgentResult(task_id="k", agent_id="knowledge", agent_role="knowledge", status="SUCCEEDED", output_summary="reference", packet=KnowledgeEvidencePacket(query="q").model_dump(), trace_id="k-trace")
    result = reviewer.review(task, [knowledge], requires_project_data=True)
    assert result.packet["status"] == "CONFLICT"
    assert not (DATA_TOOLS & KNOWLEDGE_TOOLS) and not (DRAWING_TOOLS & KNOWLEDGE_TOOLS)
    assert HVAC_TOOLS == REVIEWER_TOOLS == frozenset()
    with pytest.raises(PermissionError):
        RoleToolExecutor(context, KNOWLEDGE_TOOLS).call_project("get_equipment_kpis", "project-a")


def test_single_and_multi_factory_compatibility(tmp_path):
    _, _, _, context = runtime(tmp_path)
    assert isinstance(create_agent_runtime(context, "multi"), MultiAgentRuntime)
    assert isinstance(create_agent_runtime(context, "single"), AgentRuntime)
