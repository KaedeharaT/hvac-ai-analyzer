"""Frozen research-protocol manifests and repeat-run aggregation.

The public code stores only hashes and anonymised project identifiers.  Raw
data, annotations, model weights, and trace databases remain outside Git.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any


SPLITS = frozenset({"development", "validation", "frozen_test", "external_test"})


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frozen_manifest(path: str | Path, expected_type: str) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("registry_type") != expected_type or not payload.get("version") or payload.get("frozen") is not True:
        raise ValueError(f"{expected_type} registry must declare a version and frozen=true")
    return {"path": str(source.resolve()), "sha256": sha256_file(source), "version": str(payload["version"]), "payload": payload}


def load_split_registry(path: str | Path, *, project_id: str, dataset_sha256: str) -> dict[str, Any]:
    registry = _frozen_manifest(path, "project_split")
    matches = [item for item in registry["payload"].get("assignments", []) if item.get("project_id") == project_id]
    if len(matches) != 1 or matches[0].get("split") not in SPLITS:
        raise ValueError(f"Frozen split registry must assign {project_id} exactly once")
    declared_hash = matches[0].get("dataset_sha256")
    if declared_hash and declared_hash != dataset_sha256:
        raise ValueError("Dataset SHA-256 does not match its frozen project-split assignment")
    return {key: value for key, value in registry.items() if key != "payload"} | {"split": matches[0]["split"]}


def load_gt_freeze(path: str | Path, *, ground_truth_sha256: str) -> dict[str, Any]:
    registry = _frozen_manifest(path, "ground_truth")
    if registry["payload"].get("ground_truth_sha256") != ground_truth_sha256:
        raise ValueError("Ground-truth CSV SHA-256 does not match its frozen GT manifest")
    return {key: value for key, value in registry.items() if key != "payload"}


def load_evaluation_freeze(path: str | Path) -> dict[str, Any]:
    registry = _frozen_manifest(path, "evaluation_dataset")
    return {key: value for key, value in registry.items() if key != "payload"}


def cv_provenance(config: dict[str, Any]) -> dict[str, Any] | None:
    model = config.get("cv_model_path")
    split = config.get("cv_split_registry_path")
    if not model and not split:
        return None
    if not model or not split:
        raise ValueError("CV provenance requires both cv_model_path and cv_split_registry_path")
    path = Path(model)
    if not path.is_file():
        raise FileNotFoundError(f"CV model does not exist: {path}")
    registry = _frozen_manifest(split, "cv_split")
    return {
        "model_id": config.get("detector_model_id"), "model_sha256": sha256_file(path),
        "classes": list(config.get("detector_classes", [])),
        "confidence_threshold": config.get("detector_confidence_threshold"),
        "image_size": config.get("detector_image_size"),
        "split_registry": {key: value for key, value in registry.items() if key != "payload"},
    }


def aggregate_repetitions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate numeric metrics without pretending a single run is stable."""
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values[key].append(float(value))
    return {
        "repeat_count": len(rows),
        "metrics": {
            key: {"mean": statistics.mean(items), "std": statistics.stdev(items) if len(items) > 1 else 0.0,
                  "min": min(items), "max": max(items), "n": len(items)}
            for key, items in sorted(values.items())
        },
    }


def export_agent_traces(traces: list[dict[str, Any]], destination: str | Path, *, experiment_id: str) -> dict[str, Any]:
    target = Path(destination); target.parent.mkdir(parents=True, exist_ok=True)
    fields = ("trace_id", "parent_trace_id", "project_id", "conversation_id", "intent", "agent_architecture", "plan", "tool_calls", "evidence_checks", "reflections", "memory_used", "knowledge_sources", "llm_calls", "status", "grounded", "abstained", "multi_agent")
    with target.open("x", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps({"experiment_id": experiment_id, **{key: trace.get(key) for key in fields}}, ensure_ascii=False, default=str) + "\n")
    return {"path": str(target), "sha256": sha256_file(target), "trace_count": len(traces)}
