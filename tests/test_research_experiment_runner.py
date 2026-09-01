from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from building_ai.research.experiment_runner import ResearchExperimentRunner, sha256_file


FIXTURE = Path(__file__).parent / "fixtures" / "ahp_group_smoke.csv"


def config() -> dict:
    return {
        "experiment_name": "deterministic-smoke", "project_id": "P01",
        "dataset_path": str(FIXTURE.resolve()), "semantic_backend": "offline",
        "seed": 0, "timezone": "Asia/Tokyo", "finalize": True,
        "diagnosis": {"low_cop_threshold": 2.5, "low_delta_t_threshold_c": 3.0, "low_load_threshold": 0.3, "min_samples": 3},
    }


def clean_git(*_):
    return {"git_commit": "a" * 40, "git_dirty": False, "git_available": True}


def test_hash_is_content_addressed():
    assert len(sha256_file(FIXTURE)) == 64
    assert sha256_file(FIXTURE) == sha256_file(FIXTURE)


def test_experiment_snapshots_provenance_exports_and_is_immutable(tmp_path, monkeypatch):
    monkeypatch.setattr("building_ai.research.experiment_runner.git_metadata", clean_git)
    runner = ResearchExperimentRunner(tmp_path / "experiments")
    result = runner.run(config(), experiment_id="EXP-ONE")
    root = tmp_path / "experiments" / "EXP-ONE"
    assert result["status"] == "FINALIZED"
    for name in ("config.json", "dataset_manifest.json", "git.json", "environment.json", "semantic_mapping.csv", "equipment_mapping.csv", "kpi_summary.csv", "kpi_timeseries.csv", "energy_summary.json", "diagnostics.json", "results.json", "manifest.json", "reproduce.json"):
        assert (root / name).is_file()
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sha256"] == sha256_file(FIXTURE) and manifest["data_revision"].startswith("r-")
    snapshot = json.loads((root / "config.json").read_text(encoding="utf-8"))
    assert snapshot["git_commit"] == "a" * 40 and snapshot["semantic_mapping_version"].startswith("sem-v")
    with pytest.raises(FileExistsError):
        runner.run(config(), experiment_id="EXP-ONE")


def test_same_dataset_config_and_seed_reproduce_deterministic_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr("building_ai.research.experiment_runner.git_metadata", clean_git)
    runner = ResearchExperimentRunner(tmp_path / "experiments")
    runner.run(config(), experiment_id="EXP-ONE")
    runner.run(config(), experiment_id="EXP-TWO")
    first, second = tmp_path / "experiments" / "EXP-ONE", tmp_path / "experiments" / "EXP-TWO"
    assert (first / "semantic_mapping.csv").read_bytes() == (second / "semantic_mapping.csv").read_bytes()
    assert json.loads((first / "energy_summary.json").read_text(encoding="utf-8")) == json.loads((second / "energy_summary.json").read_text(encoding="utf-8"))
    replay = ResearchExperimentRunner.load_config(first / "config.json")
    assert replay["dataset_path"] == str(FIXTURE.resolve()) and replay["seed"] == 0


def test_ground_truth_is_separate_and_metrics_are_exported(tmp_path, monkeypatch):
    monkeypatch.setattr("building_ai.research.experiment_runner.git_metadata", clean_git)
    gt = tmp_path / "gt.csv"
    pd.DataFrame({"point_name": ["AHP#3-1 往水温度 (°C)"], "ground_truth_label": ["heat_source_supply_temp"], "annotator": ["researcher"], "annotation_date": ["2026-09-01"]}).to_csv(gt, index=False)
    payload = config(); payload["ground_truth_path"] = str(gt)
    runner = ResearchExperimentRunner(tmp_path / "experiments")
    runner.run(payload, experiment_id="EXP-GT")
    root = tmp_path / "experiments" / "EXP-GT"
    metrics = json.loads((root / "semantic_metrics.json").read_text(encoding="utf-8"))
    snapshot = json.loads((root / "config.json").read_text(encoding="utf-8"))
    assert metrics["n"] == 1 and snapshot["ground_truth"]["ground_truth_version"].startswith("gt-v")


def test_versioned_mapping_override_preserves_ai_prediction(tmp_path, monkeypatch):
    monkeypatch.setattr("building_ai.research.experiment_runner.git_metadata", clean_git)
    mapping = tmp_path / "mapping.csv"
    pd.DataFrame({"point_name": ["AHP#3-1 往水温度 (°C)"], "final_label": ["other"], "annotator": ["researcher"], "annotation_date": ["2026-09-01"]}).to_csv(mapping, index=False)
    payload = config(); payload["mapping_overrides_path"] = str(mapping)
    runner = ResearchExperimentRunner(tmp_path / "experiments")
    runner.run(payload, experiment_id="EXP-MAPPING")
    rows = pd.read_csv(tmp_path / "experiments" / "EXP-MAPPING" / "semantic_mapping.csv")
    item = rows.loc[rows["raw_name"] == "AHP#3-1 往水温度 (°C)"].iloc[0]
    snapshot = json.loads((tmp_path / "experiments" / "EXP-MAPPING" / "config.json").read_text(encoding="utf-8"))
    assert item["final_analysis_mapping"] == "other" and item["ai_prediction"] != "other"
    assert snapshot["mapping_override"]["mapping_override_version"].startswith("map-v")


def test_strict_mode_rejects_dirty_final_and_allows_explicit_draft(tmp_path, monkeypatch):
    monkeypatch.setattr("building_ai.research.experiment_runner.git_metadata", lambda *_: {"git_commit": "b" * 40, "git_dirty": True, "git_available": True})
    runner = ResearchExperimentRunner(tmp_path / "experiments")
    with pytest.raises(RuntimeError):
        runner.run(config(), experiment_id="EXP-DIRTY")
    draft = runner.run(config(), experiment_id="EXP-DRAFT", allow_dirty=True)
    assert draft["status"] == "DRAFT"


def test_multi_agent_architecture_is_snapshotted_for_research_comparison(tmp_path, monkeypatch):
    monkeypatch.setattr("building_ai.research.experiment_runner.git_metadata", clean_git)
    payload = config(); payload["agent_mode"] = "multi"
    runner = ResearchExperimentRunner(tmp_path / "experiments")
    runner.run(payload, experiment_id="EXP-MULTI")
    snapshot = json.loads((tmp_path / "experiments" / "EXP-MULTI" / "config.json").read_text(encoding="utf-8"))
    architecture = snapshot["agent_architecture"]
    assert architecture["mode"] == "multi"
    assert architecture["version"] == "multi-v1"
    assert architecture["tool_permissions"] == "read_only"


def test_publication_pipeline_exports_vectors_without_requiring_every_kpi(tmp_path, monkeypatch):
    monkeypatch.setattr("building_ai.research.experiment_runner.git_metadata", clean_git)
    runner = ResearchExperimentRunner(tmp_path / "experiments")
    runner.run(config(), experiment_id="EXP-FIGURE")
    root = tmp_path / "experiments" / "EXP-FIGURE"
    script = Path(__file__).parents[1] / "scripts" / "generate_paper_figures.py"
    result = subprocess.run([sys.executable, str(script), str(root)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (root / "figures" / "figure_energy_trend.pdf").is_file()
