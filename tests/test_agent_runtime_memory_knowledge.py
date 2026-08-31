from types import SimpleNamespace

from building_ai.agent_runtime import AgentRuntime
from building_ai.knowledge import KnowledgeService
from building_ai.knowledge_catalog import curated_facts, source_registry
from building_ai.memory import MemoryStore
from building_ai.observability import TraceStore
from building_ai.storage import Database


class _Tools:
    def call(self, name, **kwargs):
        if name == 'get_equipment_drawing_location':
            equipment_id = kwargs.get('equipment_id')
            return SimpleNamespace(ok=True, data={
                'equipment_id': equipment_id, 'reliable': equipment_id == 'CH-01',
                'locations': ([{'file_name': 'plan.png', 'page_number': 1, 'reviewed_class': 'aircon', 'class_name': 'aircon'}] if equipment_id == 'CH-01' else []),
            }, error=None)
        payload = [] if name == 'get_diagnostic_findings' else {'tool': name, **kwargs}
        return SimpleNamespace(ok=True, data=payload, error=None)


class _Controller:
    def __init__(self): self.questions=[]
    def _equipment_name(self, question):
        return next((name for name in ('CH-01', 'HP-02') if name.lower() in question.lower()), None)
    def _grounded_data_answer(self, question, equipment, grounded, evidence):
        self.questions.append(question); return f'grounded {equipment or "project"}'
    def answer(self, question, **kwargs):
        self.questions.append(question); return 'general guidance'


def _runtime(tmp_path):
    database=Database(tmp_path/'agent.sqlite3'); controller=_Controller()
    context=SimpleNamespace(database=database, agent=SimpleNamespace(tools=_Tools()), agent_controller=controller, ensure_project_loaded=lambda _:None)
    return AgentRuntime(context), controller, database


def test_conversation_memory_resolves_pronoun_in_real_runtime_trace(tmp_path):
    runtime, controller, database=_runtime(tmp_path)
    runtime.run('CH-01 equipment efficiency', 'project-a', 'chat-a')
    followup=runtime.run('what is its efficiency?', 'project-a', 'chat-a')
    trace=TraceStore(database).get(followup.trace_id)
    assert controller.questions[-1] == 'what is CH-01 efficiency?'
    assert trace['memory_used'][0]['summary']['equipment_id'] == 'CH-01'


def test_project_memory_is_scoped_and_cannot_leak_between_projects(tmp_path):
    runtime, _, database=_runtime(tmp_path)
    runtime.run('CH-01 COP', 'project-a', 'chat-a')
    runtime.run('HP-02 COP', 'project-b', 'chat-b')
    memory=MemoryStore(database)
    assert memory.get_project('project-a', 'last_analysis')['intent'] == 'equipment_analysis'
    assert memory.get('project-a', 'chat-b', 'focus', 'equipment') is None
    assert memory.get_project('project-a', 'last_analysis') != memory.get_project('project-b', 'last_analysis')


def test_knowledge_ingestion_chunks_retrieves_cited_runtime_source(tmp_path):
    runtime, _, database=_runtime(tmp_path)
    ids=KnowledgeService(database).ingest(
        'Pump Operations Guide', 'https://example.invalid/pump-guide',
        'Inspect the chilled water pump bypass valve when differential pressure is unstable. ' * 4,
        'Pump control', {'authority': 'internal'}, chunk_size=100,
    )
    result=runtime.run('HVAC pump bypass valve guidance')
    trace=TraceStore(database).get(result.trace_id)
    assert len(ids) > 1
    assert result.sources and result.sources[0]['citation'].startswith('Pump Operations Guide')
    assert result.sources[0]['metadata'] == {'authority': 'internal'}
    assert trace['knowledge_sources'] == result.sources


def test_project_recommendation_uses_project_tools_and_real_catalog_sources(tmp_path):
    runtime, _, database = _runtime(tmp_path)
    KnowledgeService(database).ingest_catalog(source_registry(), curated_facts())
    result = runtime.run('CH-01 why should I improve it first?', 'project-a', 'chat-a')
    trace = TraceStore(database).get(result.trace_id)
    assert trace['intent'] == 'recommendation'
    assert {'get_energy_opportunities', 'get_diagnostic_findings', 'get_analysis_results', 'search_knowledge'} <= {item['tool'] for item in trace['tool_calls']}
    assert result.sources
    assert result.sources[0]['citation']


def test_drawing_question_uses_read_only_location_tool_and_abstains_when_unmapped(tmp_path):
    runtime, _, database = _runtime(tmp_path)
    found = runtime.run('Where is CH-01 on the drawing?', 'project-a', 'drawing-a')
    missing = runtime.run('Where is HP-02 on the drawing?', 'project-a', 'drawing-b')
    assert found.tools_used == ['get_equipment_drawing_location']
    assert 'plan.png' in found.answer and not found.abstained
    assert missing.tools_used == ['get_equipment_drawing_location']
    assert missing.abstained
    assert TraceStore(database).get(missing.trace_id)['intent'] == 'drawing_query'
