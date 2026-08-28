from building_ai.config import Settings
from building_ai.ui.context import ApplicationContext


class _Provider:
    is_configured = True

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append((prompt, kwargs))
        return "ABSTAIN: 工具证据已查询。"


class _Manager:
    def __init__(self, provider):
        self.provider = provider

    def get_provider(self):
        return self.provider


def test_agent_controller_uses_current_project_tools_and_hides_internal_abstain(tmp_path):
    context = ApplicationContext(Settings(provider="not_configured", data_dir=tmp_path / "data"))
    project = context.projects.create("Current project")
    context.open_project(project.project_id)
    provider = _Provider()
    context.llm_manager = _Manager(provider)

    events = []
    response = context.agent_controller.answer(
        "AHP-3-1 有什么问题？", lambda tool, state: events.append((tool, state))
    )

    assert response == "Analysis has not been run for the current project."
    assert ("get_equipment_summary", "SUCCESS") in events
    assert ("get_equipment_kpis", "SUCCESS") in events
    assert ("get_diagnostic_findings", "SUCCESS") in events
    assert not provider.prompts


def test_identity_question_uses_no_project_tools(tmp_path):
    context = ApplicationContext(Settings(provider="not_configured", data_dir=tmp_path / "data"))
    provider = _Provider(); context.llm_manager = _Manager(provider)
    events = []
    response = context.agent_controller.answer("你是谁？", lambda tool, state: events.append((tool, state)))
    assert response
    assert events == []
    assert provider.prompts


def test_general_hvac_question_is_not_blocked_by_project_evidence(tmp_path):
    context = ApplicationContext(Settings(provider="not_configured", data_dir=tmp_path / "data"))
    response = context.agent_controller.answer("冷冻水 ΔT 偏低一般是什么原因？")
    assert "general HVAC engineering guidance" in response
    assert "flow" in response


def test_energy_saving_request_without_project_evidence_is_general_guidance(tmp_path):
    context = ApplicationContext(Settings(provider="not_configured", data_dir=tmp_path / "data"))
    project = context.projects.create("Empty")
    context.open_project(project.project_id)
    events = []
    response = context.agent_controller.answer("给我节能方案", lambda tool, state: events.append((tool, state)))
    assert ("get_energy_opportunities", "SUCCESS") in events
    assert "general HVAC engineering guidance" in response
