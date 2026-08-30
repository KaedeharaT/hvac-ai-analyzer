"""Run the real Local-Qwen BuildingAI E2E evaluation."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from building_ai.e2e_evaluation import run_e2e_evaluation
parser=argparse.ArgumentParser(); group=parser.add_mutually_exclusive_group(); group.add_argument('--quick',action='store_true'); group.add_argument('--full',action='store_true'); parser.add_argument('--round',default='full')
args=parser.parse_args(); result=run_e2e_evaluation(quick=args.quick,round_label=args.round,output_dir=ROOT/'artifacts'/'evaluation'/'e2e')
print(json.dumps({key:result.get(key) for key in ('evaluation_type','runner_successful','status','reason','provider','model','case_count','metrics','failure_categories')},ensure_ascii=False,indent=2))
raise SystemExit(0 if result.get('runner_successful') else 2)
