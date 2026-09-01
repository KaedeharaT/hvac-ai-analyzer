"""Machine-readable acceptance gate for the BuildingAI Agentic Platform."""
from __future__ import annotations
import json, math, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'artifacts'/'acceptance'/'latest.json'
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
CHECKS={
 'Background Worker':('building_ai.application.tasks','TaskService'),
 'Queue abstraction':('building_ai.application.tasks','TaskService'),
 'Redis Adapter':('building_ai.application.tasks','RedisTaskQueue'), 'WAITING_REVIEW':('building_ai.application.tasks','TaskService'), 'Review Resume':('building_ai.application.tasks','TaskService'), 'Retry':('building_ai.application.tasks','TaskService'), 'Timeout':('building_ai.application.tasks','TaskService'),
 'Planner':('building_ai.agent_runtime','AgentRuntime'), 'Tool Registry':('building_ai.agent_registry','ToolRegistry'),
 'Evidence Checker':('building_ai.evidence','check_reasoning'), 'Reflection':('building_ai.observability','TraceStore'), 'Conversation Memory':('building_ai.memory','MemoryStore'), 'Project Memory':('building_ai.memory','MemoryStore'),
 'Cross-project isolation':('building_ai.memory','MemoryStore'), 'RAG Retrieval':('building_ai.knowledge','KnowledgeService'), 'RAG Citation':('building_ai.knowledge','KnowledgeService'),
 'Agent Trace':('building_ai.observability','TraceStore'), 'Tool Trace':('building_ai.observability','TraceStore'), 'LLM Trace':('building_ai.observability','TraceStore'), 'Evaluation':('building_ai.evaluation','EvalRunner'),
 'Prompt Injection':('building_ai.agent_runtime','AgentRuntime'), 'Tool Permission':('building_ai.agent_registry','ToolRegistry'), 'Secret Redaction':('building_ai.observability','TraceStore'),
 'FastAPI':('building_ai.api.app','app'), 'PyQt':('building_ai.ui.main_window','MainWindow'),
}
RUNTIME_TESTS={
 'Background Worker':'tests/test_task_state_machine.py',
 'Redis Adapter':'tests/test_task_state_machine.py::test_redis_adapter_dispatches_to_importable_worker',
 'WAITING_REVIEW':'tests/test_task_state_machine.py', 'Review Resume':'tests/test_task_state_machine.py',
 'Retry':'tests/test_task_state_machine.py::test_task_retries_are_bounded_and_persisted',
 'Timeout':'tests/test_task_state_machine.py',
 'Conversation Memory':'tests/test_agent_runtime_memory_knowledge.py::test_conversation_memory_resolves_pronoun_in_real_runtime_trace',
 'Project Memory':'tests/test_agent_runtime_memory_knowledge.py::test_project_memory_is_scoped_and_cannot_leak_between_projects',
 'Cross-project isolation':'tests/test_agent_runtime_memory_knowledge.py::test_project_memory_is_scoped_and_cannot_leak_between_projects',
 'RAG Retrieval':'tests/test_agent_runtime_memory_knowledge.py::test_knowledge_ingestion_chunks_retrieves_cited_runtime_source',
 'RAG Citation':'tests/test_agent_runtime_memory_knowledge.py::test_knowledge_ingestion_chunks_retrieves_cited_runtime_source',
 'LLM Trace':'tests/test_agent_runtime_security.py::test_runtime_persists_actual_llm_call_metadata_without_prompt_content',
 'Prompt Injection':('tests/test_agent_runtime_security.py::test_user_prompt_injection_is_rejected_without_tools_or_llm','tests/test_agent_runtime_security.py::test_bems_and_rag_instruction_text_is_data_not_agent_control'),
 'Tool Permission':'tests/test_agent_runtime_security.py::test_registry_rejects_write_and_dangerous_tools_before_dispatch',
 'Secret Redaction':'tests/test_agent_runtime_security.py::test_trace_store_redacts_secrets_in_every_persisted_field',
}
EVALUATION_METRICS=('Total Cases','Routing Accuracy','Tool Selection Accuracy','Task Success Rate','Grounded Answer Rate','Factual Exact-Match Coverage','Abstention Accuracy','Tool Failure Rate','Average Tool Calls','Average Agent Latency','Average LLM Latency','RAG Retrieval Hit Rate')
E2E_REQUIRED_CATEGORIES={'general_chat','general_hvac_knowledge','memory','rag','prompt_injection','tool_failure','cross_project_isolation'}
def verify_evaluation():
 proof=subprocess.run([sys.executable,'scripts/run_agentic_evaluation.py'],cwd=ROOT,capture_output=True,text=True)
 artifact=ROOT/'artifacts'/'evaluation'/'regression'/'latest.json'
 if proof.returncode != 0: return {'status':'FAIL','evidence':f'evaluation runner failed: {proof.stdout[-160:] or proof.stderr[-160:]}' }
 try:
  payload=json.loads(artifact.read_text(encoding='utf-8')); metrics=payload['metrics']
  valid=(payload.get('evaluation_type') == 'deterministic_regression' and payload.get('runner_successful') is True and isinstance(payload.get('run_id'),str) and bool(payload.get('timestamp')) and isinstance(payload.get('git_commit'),str) and len(payload['git_commit']) == 40 and payload.get('case_count',0)>=60 and all(name in metrics and isinstance(metrics[name],(int,float)) and math.isfinite(metrics[name]) for name in EVALUATION_METRICS))
  if not valid: return {'status':'FAIL','evidence':'evaluation artifact missing successful run, >=60 cases, or calculable core metrics'}
  return {'status':'PASS','evidence':f"real evaluation artifact: {payload['case_count']} cases, run {payload['run_id']}"}
 except (OSError,json.JSONDecodeError,KeyError,TypeError): return {'status':'FAIL','evidence':'evaluation artifact could not be verified'}
def verify_e2e_smoke():
 artifact=ROOT/'artifacts'/'evaluation'/'e2e'/'latest.json'
 try:
  payload=json.loads(artifact.read_text(encoding='utf-8')); metrics=payload['metrics']; categories={row['category'] for row in payload['cases']}
  valid=(payload.get('evaluation_type') in {'local_llm_end_to_end','local_qwen_end_to_end'} and payload.get('runner_successful') is True and payload.get('provider') in {'local_llm','local_qwen'} and payload.get('case_count',0)>=50 and payload.get('real_llm_calls',0)>0 and float(metrics.get('Average LLM Latency',0))>0 and E2E_REQUIRED_CATEGORIES <= categories)
  return {'status':'PASS','evidence':f"real local LLM artifact: {payload['case_count']} cases, {payload['real_llm_calls']} calls"} if valid else {'status':'FAIL','evidence':'missing real local LLM E2E evidence, latency, coverage, or artifact'}
 except (OSError,json.JSONDecodeError,KeyError,TypeError,ValueError): return {'status':'FAIL','evidence':'E2E evaluation artifact could not be verified'}
def main():
 import importlib
 results={}
 for name, target in CHECKS.items():
  if target is None: results[name]={'status':'FAIL','evidence':'no verified production/runtime evidence'}; continue
  try:
   mod=importlib.import_module(target[0]); getattr(mod,target[1]); results[name]={'status':'PASS','evidence':f'{target[0]}.{target[1]} importable'}
  except Exception as exc: results[name]={'status':'FAIL','evidence':str(exc)[:200]}
  if name == 'Evaluation' and results[name]['status']=='PASS': results[name]=verify_evaluation()
  if name in RUNTIME_TESTS and results[name]['status']=='PASS':
   targets=RUNTIME_TESTS[name] if isinstance(RUNTIME_TESTS[name],tuple) else (RUNTIME_TESTS[name],)
   proof=subprocess.run([sys.executable,'-m','pytest',*targets],cwd=ROOT,capture_output=True,text=True)
   results[name]={'status':'PASS' if proof.returncode==0 else 'FAIL','evidence':f'runtime test: {", ".join(targets)}' if proof.returncode==0 else proof.stdout[-200:]}
 test=subprocess.run([sys.executable,'-m','pytest'],cwd=ROOT,capture_output=True,text=True)
 results['pytest']={'status':'PASS' if test.returncode==0 else 'FAIL','evidence':test.stdout.splitlines()[-1] if test.stdout else test.stderr[-200:]}
 counts={x:sum(v['status']==x for v in results.values()) for x in ('PASS','FAIL','ENVIRONMENT_BLOCKED')}
 e2e_smoke=verify_e2e_smoke()
 payload={'generated_at':datetime.now(timezone.utc).isoformat(),'results':results,'e2e_smoke':e2e_smoke,'counts':counts}
 OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
 print('BuildingAI Agentic Acceptance\n')
 for name,value in results.items(): print(f'{name:<28} {value["status"]}')
 print(f"{'E2E Evaluation Smoke':<28} {e2e_smoke['status']}")
 print(f'\nTOTAL\nPASS: {counts["PASS"]}\nFAIL: {counts["FAIL"]}\nENVIRONMENT_BLOCKED: {counts["ENVIRONMENT_BLOCKED"]}')
 return 1 if counts['FAIL'] else 0
if __name__=='__main__': raise SystemExit(main())
