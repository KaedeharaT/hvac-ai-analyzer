"""Run the unchanged deterministic Agent regression through Multi-Agent mode."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from building_ai.evaluation import run_deterministic_multi_agent_evaluation


if __name__ == "__main__":
    result = run_deterministic_multi_agent_evaluation(ROOT / "artifacts" / "evaluation" / "multi_agent_regression" / "latest.json")
    print(json.dumps({"run_id": result["run_id"], "case_count": result["case_count"], "metrics": result["metrics"],
                      "failed_cases": [row["case_id"] for row in result["failed_cases"]]}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["runner_successful"] else 1)
