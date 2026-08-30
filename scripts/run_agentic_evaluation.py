"""Run BuildingAI's deterministic agent evaluation independently of pytest."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from building_ai.evaluation import run_deterministic_evaluation


if __name__ == "__main__":
    result = run_deterministic_evaluation(ROOT / "artifacts" / "evaluation" / "regression" / "latest.json")
    print(json.dumps({
        "run_id": result["run_id"], "case_count": result["case_count"],
        "metrics": result["metrics"], "failure_analysis": result["failure_analysis"],
        "failed_cases": [row["case_id"] for row in result["failed_cases"]],
    }, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["runner_successful"] else 1)
