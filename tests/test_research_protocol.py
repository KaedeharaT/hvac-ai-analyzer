from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from building_ai.research.experiment_runner import ResearchExperimentRunner, sha256_file
from building_ai.research.matrix_runner import run_matrix
from building_ai.research.agent_evaluation import run_repeated_agent_evaluation
from building_ai.research.protocol import aggregate_repetitions, export_agent_traces


FIXTURE = Path(__file__).parent / "fixtures" / "ahp_group_smoke.csv"


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _config(tmp_path: Path, split: Path, gt: Path, freeze: Path) -> dict:
    return {"experiment_name": "formal-protocol", "project_id": "P01", "dataset_path": str(FIXTURE.resolve()),
            "semantic_backend": "offline", "seed": 0, "timezone": "Asia/Tokyo", "finalize": True,
            "research_protocol": "formal", "split_registry_path": str(split), "ground_truth_path": str(gt),
            "ground_truth_freeze_path": str(freeze), "diagnosis": {}}


def test_formal_run_requires_matching_frozen_split_and_gt(tmp_path, monkeypatch):
    monkeypatch.setattr("building_ai.research.experiment_runner.git_metadata", lambda *_: {"git_commit": "a" * 40, "git_dirty": False, "git_available": True})
    gt = tmp_path / "gt.csv"
    pd.DataFrame({"point_name": ["AHP#3-1 往水温度 (°C)"], "ground_truth_label": ["heat_source_supply_temp"]}).to_csv(gt, index=False)
    split = _write(tmp_path / "split.json", {"registry_type": "project_split", "version": "split-v1", "frozen": True,
                                                 "assignments": [{"project_id": "P01", "split": "frozen_test", "dataset_sha256": sha256_file(FIXTURE)}]})
    freeze = _write(tmp_path / "gt_freeze.json", {"registry_type": "ground_truth", "version": "gt-v1", "frozen": True,
                                                     "ground_truth_sha256": sha256_file(gt)})
    result = ResearchExperimentRunner(tmp_path / "artifacts").run(_config(tmp_path, split, gt, freeze), experiment_id="EXP-FORMAL")
    assert result["protocol"]["split"]["split"] == "frozen_test"
    assert result["protocol"]["ground_truth_freeze"]["version"] == "gt-v1"


def test_formal_run_rejects_missing_freezes_and_exports_trace(tmp_path, monkeypatch):
    monkeypatch.setattr("building_ai.research.experiment_runner.git_metadata", lambda *_: {"git_commit": "a" * 40, "git_dirty": False, "git_available": True})
    with pytest.raises(ValueError):
        ResearchExperimentRunner(tmp_path / "artifacts").run({"experiment_name": "bad", "project_id": "P01", "dataset_path": str(FIXTURE), "research_protocol": "formal"}, experiment_id="EXP-BAD")
    exported = export_agent_traces([{"trace_id": "t1", "answer": "excluded", "tool_calls": []}], tmp_path / "traces.jsonl", experiment_id="EXP-TRACE")
    assert exported["trace_count"] == 1 and "answer" not in (tmp_path / "traces.jsonl").read_text(encoding="utf-8")


def test_repeat_aggregation_reports_mean_std_and_n():
    result = aggregate_repetitions([{"accuracy": .8, "latency_ms": 10}, {"accuracy": .9, "latency_ms": 20}])
    assert result["repeat_count"] == 2
    accuracy = result["metrics"]["accuracy"]
    assert accuracy["mean"] == pytest.approx(.85) and accuracy["std"] == pytest.approx(.070710678)
    assert accuracy["min"] == .8 and accuracy["max"] == .9 and accuracy["n"] == 2


def test_config_matrix_runs_declared_baselines_without_source_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr("building_ai.research.experiment_runner.git_metadata", lambda *_: {"git_commit": "a" * 40, "git_dirty": False, "git_available": True})
    config = _write(tmp_path / "baseline.json", {"experiment_name": "baseline", "project_id": "P01", "dataset_path": str(FIXTURE.resolve()), "semantic_backend": "offline", "seed": 0, "finalize": True})
    matrix = _write(tmp_path / "matrix.json", {"matrix_version": "research-matrix-v1", "runs": [{"name": "keyword-baseline", "config": config.name, "seeds": [0, 1]}]})
    result = run_matrix(matrix, artifact_root=tmp_path / "artifacts")
    assert result["aggregates"]["keyword-baseline"]["repeat_count"] == 2
    assert len(result["runs"]) == 2


def test_repeated_agent_evaluation_aggregates_independent_runs(tmp_path, monkeypatch):
    counter = iter(("r1", "r2"))
    monkeypatch.setattr("building_ai.research.agent_evaluation.run_e2e_evaluation", lambda **_: {"runner_successful": True, "run_id": next(counter), "metrics": {"Task Success Rate": 1.0, "Average Agent Latency": 12.0}})
    result = run_repeated_agent_evaluation(repeats=2, output_dir=tmp_path / "repeats")
    assert result["aggregate"]["metrics"]["Average Agent Latency"]["n"] == 2
