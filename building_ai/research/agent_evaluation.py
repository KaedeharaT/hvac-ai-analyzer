"""Repeated Local-LLM evaluation with immutable aggregate artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from building_ai.e2e_evaluation import run_e2e_evaluation
from building_ai.research.protocol import aggregate_repetitions, sha256_file


def run_repeated_agent_evaluation(*, repeats: int, output_dir: str | Path, quick: bool = False) -> dict[str, Any]:
    if repeats < 2:
        raise ValueError("Repeated LLM evaluation requires repeats >= 2")
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"Repeated evaluation artifact is immutable: {root}")
    root.mkdir(parents=True)
    runs: list[dict[str, Any]] = []
    for index in range(repeats):
        payload = run_e2e_evaluation(quick=quick, round_label=f"repeat-{index + 1}")
        if not payload.get("runner_successful"):
            raise RuntimeError(f"Local-LLM repeat {index + 1} did not complete: {payload.get('reason', 'unknown')}")
        (root / f"run-{index + 1:02d}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        runs.append({"run_id": payload["run_id"], **payload["metrics"]})
    aggregate = {"evaluation_version": "agent-e2e-repeat-v1", "repeat_count": repeats,
                 "runs": runs, "aggregate": aggregate_repetitions(runs)}
    target = root / "aggregate.json"; target.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    aggregate["aggregate_sha256"] = sha256_file(target)
    return aggregate
