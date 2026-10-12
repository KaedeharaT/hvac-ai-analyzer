"""Real project/tool regressions; no LLM or fabricated tool responses."""
import pandas as pd
import pytest

from building_ai.config import Settings
from building_ai.ui.context import ApplicationContext
from building_ai.agent_runtime import AgentRuntime
from building_ai.observability import TraceStore


def make_context(tmp_path, monkeypatch):
    monkeypatch.setenv('BUILDING_AI_CONFIG_DIR', str(tmp_path / 'config'))
    context = ApplicationContext(Settings(data_dir=tmp_path / 'data'))
    project = context.projects.create('SYNTHETIC TOOL QA'); context.open_project(project.project_id)
    frame = pd.DataFrame({'time': pd.date_range('2025-01-01', periods=4, freq='h'),
                          'CH-1 LWT °C': [6.] * 4, 'CH-1 EWT °C': [11.] * 4,
                          'CH-1 Flow m3/h': [18.] * 4, 'CH-1 Power kW': [30.] * 4,
                          'CH-2 Power kW': [10.] * 4})
    path = tmp_path / 'synthetic.csv'; frame.to_csv(path, index=False)
    context.import_data(str(path)); context.run_semantics(); context.run_diagnosis()
    return context, project.project_id


def test_energy_answer_uses_selected_equipment_not_whole_building(tmp_path, monkeypatch):
    context, project = make_context(tmp_path, monkeypatch)
    runtime = AgentRuntime(context)
    answer = runtime.run('查询 CH-1 能耗', project)
    assert '120.000 kWh' in answer.answer
    assert '160.000' not in answer.answer
    trace = TraceStore(context.database).get(answer.trace_id)
    assert any(step['tool'] == 'get_energy_summary' and step['arguments']['equipment_id'] == 'CH-1' for step in trace['plan'])
    assert context.energy_analysis_result.summary['total_energy_kwh'] == pytest.approx(160)


def test_explicit_unknown_equipment_overrides_conversation_focus(tmp_path, monkeypatch):
    context, project = make_context(tmp_path, monkeypatch)
    runtime = AgentRuntime(context)
    runtime.run('CH-1 COP', project, 'chat')
    answer = runtime.run('当前项目有 CH-999 吗？', project, 'chat')
    assert 'CH-999' in answer.answer and '3.49' not in answer.answer
    trace = TraceStore(context.database).get(answer.trace_id)
    assert 'CH-1' not in trace['query']


def test_missing_rated_capacity_and_missing_signals_are_explained(tmp_path, monkeypatch):
    context, project = make_context(tmp_path, monkeypatch)
    runtime = AgentRuntime(context)
    answer = runtime.run('CH-1 额定负荷率能算吗？', project)
    assert '额定' in answer.answer and '缺少' in answer.answer
    missing = runtime.run('CH-2 COP', project)
    assert 'missing_signals' in missing.answer
    assert 'COP：0.00' not in missing.answer
    listing = runtime.run('当前项目有哪些设备？', project)
    assert 'synthetic.csv' in listing.answer and 'CH-1' in listing.answer and 'CH-2' in listing.answer
