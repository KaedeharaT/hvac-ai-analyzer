"""UI-independent, immutable research experiment artifacts.

This module deliberately reuses the product services.  It does not turn UI
state or a mutable project cache into a paper result: every run snapshots the
input hash, configuration, code state, environment and raw numerical outputs.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid
from typing import Any

import pandas as pd

from building_ai.config import Settings
from building_ai.models import SemanticResult
from building_ai.models.semantic_result import SUPPORTED_TAXONOMY
from building_ai.services.analytics_service import AnalyticsService
from building_ai.services.diagnosis_service import DiagnosisService
from building_ai.services.energy_analysis_service import EnergyAnalysisService
from building_ai.services.equipment_service import EquipmentService
from building_ai.services.import_service import ImportService
from building_ai.services.semantic_service import PIPELINE_VERSION, SemanticService
from building_ai.storage.confirmed_mapping_store import ConfirmedMappingStore
# Import after service modules: the established diagnostics module imports the
# analytics service for its typed rule inputs.
from building_ai.core.diagnostics import DiagnosisConfig
from building_ai.multi_agent_runtime import (
    AGENT_PROMPT_VERSIONS, COORDINATION_POLICY_VERSION, MAX_REPLAN_ROUNDS, MULTI_AGENT_ARCHITECTURE_VERSION,
    REVIEW_POLICY_VERSION,
)


RESEARCH_SCHEMA_VERSION = 1
DIAGNOSIS_RULESET_VERSION = "diagnosis-rules-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Not JSON serialisable: {type(value)!r}")


def git_metadata(root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        try:
            return subprocess.check_output(["git", *arguments], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"git_commit": commit, "git_dirty": bool(status), "git_available": commit is not None}


def environment_snapshot() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("pandas", "numpy", "PyQt5", "scikit-learn", "pydantic", "matplotlib", "ultralytics", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not_installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
    }


def _semantic_rows(items: list[SemanticResult]) -> list[dict[str, Any]]:
    return [{
        "point_id": item.point_id,
        "raw_name": item.raw_name,
        "ai_prediction": item.canonical_label,
        "final_analysis_mapping": item.effective_label,
        "review_status": item.review_status,
        "unit": item.unit,
        "equipment_id": item.effective_equipment_id,
        "confidence": item.confidence,
        "algorithm_version": item.algorithm_version,
        "prompt_version": item.prompt_version,
        "model_provider": item.model_provider,
        "model_name": item.model_name,
        "reason": item.reason,
    } for item in items]


def _apply_mapping_overrides(items: list[SemanticResult], path: str | Path | None) -> dict[str, Any] | None:
    """Apply a versioned research mapping without mutating production reviews."""
    if not path:
        return None
    source = Path(path)
    rows = pd.read_csv(source)
    required = {"point_name", "final_label"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Mapping override CSV missing required columns: {', '.join(sorted(missing))}")
    overrides = {str(row.point_name): row for row in rows.itertuples(index=False)}
    for item in items:
        override = overrides.get(item.raw_name)
        if override is None:
            continue
        label = str(getattr(override, "final_label"))
        if label not in SUPPORTED_TAXONOMY:
            raise ValueError(f"Mapping override has unsupported strict label: {label}")
        item.human_verified = True; item.human_label = label; item.confirmed_label = label
        equipment = getattr(override, "equipment_id", None)
        if equipment is not None and pd.notna(equipment):
            item.confirmed_equipment_id = str(equipment)
        item.confirmation_source = "research_mapping_override"
        item.human_note = str(getattr(override, "notes", "")) if hasattr(override, "notes") else None
        annotation_date = getattr(override, "annotation_date", None)
        item.confirmed_at = str(annotation_date) if annotation_date is not None and pd.notna(annotation_date) else None
    return {"path": str(source), "sha256": sha256_file(source), "rows": int(len(rows)), "mapping_override_version": f"map-v{sha256_file(source)[:12]}"}


def _semantic_metrics(rows: list[dict[str, Any]], ground_truth: pd.DataFrame | None) -> dict[str, Any] | None:
    if ground_truth is None:
        return None
    required = {"point_name", "ground_truth_label"}
    missing = required - set(ground_truth.columns)
    if missing:
        raise ValueError(f"Ground-truth CSV missing required columns: {', '.join(sorted(missing))}")
    expected = ground_truth.loc[:, ["point_name", "ground_truth_label"]].copy()
    expected["point_name"] = expected["point_name"].astype(str)
    predicted = pd.DataFrame(rows).loc[:, ["raw_name", "final_analysis_mapping"]]
    joined = expected.merge(predicted, how="left", left_on="point_name", right_on="raw_name")
    joined["final_analysis_mapping"] = joined["final_analysis_mapping"].fillna("__missing_prediction__")
    labels = sorted(set(joined["ground_truth_label"]) | set(joined["final_analysis_mapping"]))
    accuracy = float((joined["ground_truth_label"] == joined["final_analysis_mapping"]).mean()) if len(joined) else 0.0
    per_class: list[dict[str, Any]] = []
    for label in labels:
        truth = joined["ground_truth_label"] == label
        prediction = joined["final_analysis_mapping"] == label
        tp = int((truth & prediction).sum()); fp = int((~truth & prediction).sum()); fn = int((truth & ~prediction).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"label": label, "precision": precision, "recall": recall, "f1": f1, "support": int(truth.sum())})
    return {"n": int(len(joined)), "accuracy": accuracy, "macro_f1": float(sum(x["f1"] for x in per_class) / len(per_class)) if per_class else 0.0, "per_class": per_class}


class ResearchExperimentRunner:
    """Run a deterministic product pipeline and persist an immutable research record."""
    def __init__(self, artifact_root: str | Path = "artifacts/experiments", *, repository_root: str | Path | None = None):
        self.artifact_root = Path(artifact_root)
        self.repository_root = Path(repository_root) if repository_root else Path(__file__).resolve().parents[2]

    @staticmethod
    def load_config(path: str | Path) -> dict[str, Any]:
        source = Path(path)
        if source.suffix.lower() != ".json":
            raise ValueError("Research configurations must be JSON so they can be snapshotted without an optional YAML parser")
        payload = json.loads(source.read_text(encoding="utf-8"))
        # A saved experiment config contains immutable metadata plus the original
        # runnable configuration.  Accept it directly for one-command replay.
        if isinstance(payload, dict) and isinstance(payload.get("configuration"), dict):
            payload = dict(payload["configuration"])
        if not isinstance(payload, dict):
            raise ValueError("Research configuration must be a JSON object")
        if not payload.get("dataset_path"):
            raise ValueError("Research configuration requires dataset_path")
        for key in ("dataset_path", "ground_truth_path", "mapping_overrides_path"):
            value = payload.get(key)
            if value:
                candidate = Path(value)
                payload[key] = str((source.parent / candidate).resolve()) if not candidate.is_absolute() else str(candidate)
        payload.setdefault("experiment_name", source.stem)
        payload.setdefault("project_id", "research-project")
        payload.setdefault("semantic_backend", "offline")
        payload.setdefault("seed", 0)
        payload.setdefault("timezone", "unspecified")
        payload.setdefault("diagnosis", {})
        payload.setdefault("finalize", True)
        payload.setdefault("agent_mode", "single")
        if payload["agent_mode"] not in {"single", "multi"}:
            raise ValueError("Research configuration agent_mode must be 'single' or 'multi'")
        return payload

    def run(self, config: dict[str, Any], *, experiment_id: str | None = None, allow_dirty: bool = False) -> dict[str, Any]:
        config = dict(config)
        config.setdefault("agent_mode", "single")
        if config["agent_mode"] not in {"single", "multi"}:
            raise ValueError("Research configuration agent_mode must be 'single' or 'multi'")
        source = Path(str(config["dataset_path"])).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Research dataset does not exist: {source}")
        git = git_metadata(self.repository_root)
        if bool(config.get("finalize", True)) and git["git_dirty"] and not allow_dirty:
            raise RuntimeError("Strict Research Mode refuses FINALIZED experiments from a dirty working tree; commit changes or run a DRAFT explicitly")
        if git["git_dirty"] and allow_dirty:
            config["finalize"] = False
        identifier = experiment_id or f"EXP-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        root = self.artifact_root / identifier
        if root.exists():
            raise FileExistsError(f"Experiment artifact directory already exists and is immutable: {root}")
        root.mkdir(parents=True)
        try:
            return self._run_into(root, identifier, config, source, git)
        except Exception as exc:
            self._write_json(root / "config.json", config)
            self._write_json(root / "failure.json", {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc), "git": git})
            self._append_index({"experiment_id": identifier, "status": "FAILED", "artifact_path": str(root), "git_commit": git["git_commit"], "error": str(exc)})
            raise

    def _run_into(self, root: Path, identifier: str, config: dict[str, Any], source: Path, git: dict[str, Any]) -> dict[str, Any]:
        importer = ImportService(); imported = importer.load(source, config.get("sheet"))
        metadata = imported.metadata.to_dict()
        metadata["source_file"] = source.name
        dataset = {
            "dataset_id": f"dataset-{sha256_file(source)[:16]}", "data_revision": f"r-{sha256_file(source)[:12]}",
            "original_filename": source.name, "sha256": sha256_file(source), "file_size_bytes": source.stat().st_size,
            "sheet": metadata.get("sheet"), "imported_at": datetime.now(timezone.utc).isoformat(),
            "row_count": metadata["rows"], "column_count": metadata["columns"], "time_range": {"start": metadata["start"], "end": metadata["end"]},
            "sampling_interval": metadata["sampling_interval"], "timezone": config.get("timezone", "unspecified"),
            "missing_ratio": metadata["missing_ratio"], "duplicate_rows": metadata["duplicate_rows"], "data_quality": metadata,
        }
        settings = Settings(provider="not_configured", data_dir=root / "runtime")
        semantic = SemanticService(settings)
        # Research runs must not silently borrow a user's mutable production confirmations.
        semantic.confirmed_mappings = ConfirmedMappingStore(root / "research_confirmed_mappings.sqlite3")
        backend = str(config.get("semantic_backend", "offline"))
        if backend not in {"offline", "enhanced", "research", "legacy_c1c8"}:
            raise ValueError(f"Unsupported semantic_backend: {backend}")
        semantics = semantic.analyze_dataframe(imported.dataframe, source_file=source.name, sheet=metadata.get("sheet"), project_id=str(config["project_id"]), backend=backend)  # type: ignore[arg-type]
        mapping_override = _apply_mapping_overrides(semantics.semantic_results, config.get("mapping_overrides_path"))
        equipment = EquipmentService().organize(str(config["project_id"]), semantics.semantic_results)
        analytics = AnalyticsService().analyze_project(imported.dataframe, semantics, str(config["project_id"]), metadata, equipment)
        diagnosis_config = DiagnosisConfig(**dict(config.get("diagnosis") or {}))
        diagnosis = DiagnosisService(AnalyticsService(), EquipmentService(), diagnosis_config).diagnose_project(imported.dataframe, semantics, str(config["project_id"]), metadata, organization=equipment, analytics_result=analytics)
        energy = EnergyAnalysisService().analyze(imported.dataframe, semantics, str(config["project_id"]), metadata, equipment, analytics, aggregation=config.get("aggregation"))
        semantic_rows = _semantic_rows(semantics.semantic_results)
        mapping_payload = [{key: row[key] for key in ("raw_name", "final_analysis_mapping", "unit", "equipment_id", "algorithm_version", "prompt_version")} for row in semantic_rows]
        mapping_version = f"sem-v{canonical_hash(mapping_payload)[:12]}"
        gt_path = config.get("ground_truth_path")
        ground_truth = pd.read_csv(gt_path) if gt_path else None
        gt_info = None if ground_truth is None else {"path": str(Path(gt_path).resolve()), "sha256": sha256_file(gt_path), "rows": int(len(ground_truth)), "ground_truth_version": f"gt-v{sha256_file(gt_path)[:12]}"}
        metrics = _semantic_metrics(semantic_rows, ground_truth)
        snapshot = {
            "research_schema_version": RESEARCH_SCHEMA_VERSION, "experiment_id": identifier, "experiment_name": config["experiment_name"],
            "project_id": config["project_id"], "data_revision": dataset["data_revision"], "semantic_mapping_version": mapping_version,
            "git_commit": git["git_commit"], "git_dirty": git["git_dirty"],
            "semantic_algorithm_version": PIPELINE_VERSION, "diagnosis_ruleset_version": DIAGNOSIS_RULESET_VERSION,
            "configuration": config, "random_seed": int(config.get("seed", 0)), "llm": {"provider": settings.provider, "model": settings.model, "temperature": 0, "top_p": None, "max_tokens": None, "seed": int(config.get("seed", 0))},
            "agent_architecture": {"mode": config["agent_mode"], "version": "single-v1" if config["agent_mode"] == "single" else MULTI_AGENT_ARCHITECTURE_VERSION,
                "agents": ["CoordinatorAgent", "DataAnalystAgent", "HVACExpertAgent", "KnowledgeAgent", "DrawingAgent", "ReviewerAgent"] if config["agent_mode"] == "multi" else ["AgentRuntime"],
                "coordination_policy": COORDINATION_POLICY_VERSION if config["agent_mode"] == "multi" else None,
                "review_policy": REVIEW_POLICY_VERSION if config["agent_mode"] == "multi" else None,
                "max_rounds": MAX_REPLAN_ROUNDS if config["agent_mode"] == "multi" else 1,
                "prompt_versions": dict(AGENT_PROMPT_VERSIONS) if config["agent_mode"] == "multi" else {"agent_runtime": "single-v1"},
                "tool_permissions": "read_only"},
            "knowledge_base_version": self._knowledge_version(), "detector_model": config.get("detector_model"), "ground_truth": gt_info,
            "mapping_override": mapping_override,
        }
        self._write_json(root / "config.json", snapshot)
        self._write_json(root / "dataset_manifest.json", dataset)
        self._write_json(root / "git.json", git)
        self._write_json(root / "environment.json", environment_snapshot())
        pd.DataFrame(semantic_rows).to_csv(root / "semantic_mapping.csv", index=False)
        pd.DataFrame([{"equipment_id": item.equipment_id, "equipment_name": item.name, "equipment_type": item.equipment_type.value} for item in equipment.equipment]).to_csv(root / "equipment_mapping.csv", index=False)
        self._write_kpis(root, analytics)
        self._write_json(root / "energy_summary.json", energy.to_dict(include_values=True, limit=1_000_000))
        self._write_json(root / "diagnostics.json", {"findings": [x.to_dict() for x in diagnosis.findings], "skipped": diagnosis.skipped})
        if metrics is not None:
            self._write_json(root / "semantic_metrics.json", metrics)
            pd.DataFrame(metrics["per_class"]).to_csv(root / "semantic_per_class.csv", index=False)
        summary = {"status": "FINALIZED" if config.get("finalize", True) else "DRAFT", "experiment_id": identifier, "dataset": dataset, "semantic_mapping_version": mapping_version, "metrics": metrics, "energy_summary": energy.summary, "finding_count": len(diagnosis.findings), "warnings": {"semantic": semantics.warnings, "analytics": analytics.skipped, "diagnosis": diagnosis.skipped, "energy": energy.warnings}}
        self._write_json(root / "results.json", summary)
        self._write_json(root / "reproduce.json", {"command": f"python scripts/run_research_experiment.py --config {root / 'config.json'}", "dataset_sha256": dataset["sha256"], "git_commit": git["git_commit"]})
        manifest = self._manifest(root)
        self._write_json(root / "manifest.json", manifest)
        self._append_index({"experiment_id": identifier, "status": summary["status"], "artifact_path": str(root), "git_commit": git["git_commit"], "data_revision": dataset["data_revision"], "semantic_mapping_version": mapping_version, "metrics": metrics})
        return summary

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json), encoding="utf-8")

    @staticmethod
    def _knowledge_version() -> dict[str, Any]:
        root = Path(__file__).resolve().parents[2] / "knowledge"
        manifest = root / "metadata" / "catalog_manifest.json"
        return {"catalog_manifest_sha256": sha256_file(manifest) if manifest.exists() else None, "source_registry_sha256": sha256_file(root / "source_registry.json")}

    @staticmethod
    def _write_kpis(root: Path, analytics: Any) -> None:
        summary_rows: list[dict[str, Any]] = []; series_rows: list[dict[str, Any]] = []
        units = {"supply_temp_c": "°C", "return_temp_c": "°C", "delta_t_c": "°C", "flow_lps": "L/s", "power_kw": "kW", "thermal_load_kw": "kW", "cop": "ratio"}
        for item in analytics.equipment_kpis:
            for metric, values in item.metric_summary.items():
                summary_rows.append({"equipment_id": item.equipment_id, "equipment_name": item.equipment_name, "status": item.status, "metric": metric, "unit": units.get(metric), **values})
            if item.cop is None or item.timestamps is None:
                continue
            values = {"supply_temp_c": item.cop.supply_temp_c, "return_temp_c": item.cop.return_temp_c, "delta_t_c": item.cop.delta_t_c, "flow_lps": item.cop.flow_lps, "power_kw": item.cop.input_power_kw, "thermal_load_kw": item.cop.cooling_load_kw, "cop": item.cop.series}
            for index, timestamp in enumerate(item.timestamps):
                row = {"equipment_id": item.equipment_id, "equipment_name": item.equipment_name, "timestamp": timestamp, "valid": bool(item.valid_mask.iloc[index]) if item.valid_mask is not None else False}
                for name, series in values.items():
                    row[name] = None if series is None or pd.isna(series.iloc[index]) else float(series.iloc[index])
                series_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(root / "kpi_summary.csv", index=False)
        pd.DataFrame(series_rows).to_csv(root / "kpi_timeseries.csv", index=False)

    @staticmethod
    def _manifest(root: Path) -> dict[str, Any]:
        rows = []
        for path in sorted(root.iterdir()):
            if path.name == "manifest.json" or not path.is_file():
                continue
            rows.append({"path": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
        return {"schema_version": RESEARCH_SCHEMA_VERSION, "created_at": datetime.now(timezone.utc).isoformat(), "artifacts": rows}

    def _append_index(self, row: dict[str, Any]) -> None:
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        with (self.artifact_root / "index.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(), **row}, ensure_ascii=False, default=_json) + "\n")
