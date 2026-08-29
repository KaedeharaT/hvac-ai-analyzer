"""Machine-readable acceptance gate for the BuildingAI Agentic Platform."""
from __future__ import annotations
import json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'artifacts'/'acceptance'/'latest.json'
CHECKS={
 'Background Worker':('building_ai.application.tasks','submit_background'),
 'Queue abstraction':('building_ai.application.tasks','TaskService'),
 'Redis Adapter':None, 'WAITING_REVIEW':None, 'Review Resume':None, 'Retry':None, 'Timeout':None,
 'Planner':('building_ai.agent_runtime','AgentRuntime'), 'Tool Registry':None,
 'Evidence Checker':None, 'Reflection':None, 'Conversation Memory':None, 'Project Memory':None,
 'Cross-project isolation':None, 'RAG Retrieval':None, 'RAG Citation':None,
 'Agent Trace':None, 'Tool Trace':None, 'LLM Trace':None, 'Evaluation':None,
 'Prompt Injection':None, 'Tool Permission':None, 'Secret Redaction':None,
 'FastAPI':('building_ai.api.app','app'), 'PyQt':('building_ai.ui.main_window','MainWindow'),
}
def main():
 import importlib
 results={}
 for name, target in CHECKS.items():
  if target is None: results[name]={'status':'FAIL','evidence':'no verified production/runtime evidence'}; continue
  try:
   mod=importlib.import_module(target[0]); getattr(mod,target[1]); results[name]={'status':'PASS','evidence':f'{target[0]}.{target[1]} importable'}
  except Exception as exc: results[name]={'status':'FAIL','evidence':str(exc)[:200]}
 test=subprocess.run([sys.executable,'-m','pytest'],cwd=ROOT,capture_output=True,text=True)
 results['pytest']={'status':'PASS' if test.returncode==0 else 'FAIL','evidence':test.stdout.splitlines()[-1] if test.stdout else test.stderr[-200:]}
 counts={x:sum(v['status']==x for v in results.values()) for x in ('PASS','FAIL','ENVIRONMENT_BLOCKED')}
 payload={'generated_at':datetime.now(timezone.utc).isoformat(),'results':results,'counts':counts}
 OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
 print('BuildingAI Agentic Acceptance\n')
 for name,value in results.items(): print(f'{name:<28} {value["status"]}')
 print(f'\nTOTAL\nPASS: {counts["PASS"]}\nFAIL: {counts["FAIL"]}\nENVIRONMENT_BLOCKED: {counts["ENVIRONMENT_BLOCKED"]}')
 return 1 if counts['FAIL'] else 0
if __name__=='__main__': raise SystemExit(main())
