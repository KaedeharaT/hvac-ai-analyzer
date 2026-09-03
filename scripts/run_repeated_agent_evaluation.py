"""Repeat real Local-LLM E2E evaluation and aggregate mean/std/latency."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from building_ai.research.agent_evaluation import run_repeated_agent_evaluation
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repeats", required=True, type=int)
parser.add_argument("--output-dir", required=True, type=Path)
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()
print(json.dumps(run_repeated_agent_evaluation(repeats=args.repeats, output_dir=args.output_dir, quick=args.quick), ensure_ascii=False, indent=2, default=str))
