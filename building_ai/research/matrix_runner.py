"""Configuration-driven baseline/ablation experiment matrix."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from building_ai.research.experiment_runner import ResearchExperimentRunner
from building_ai.research.protocol import aggregate_repetitions, sha256_file


def run_matrix(path: str | Path, *, artifact_root: str | Path, allow_dirty: bool = False) -> dict[str, Any]:
    source = Path(path); matrix = json.loads(source.read_text(encoding="utf-8"))
    if matrix.get("matrix_version") != "research-matrix-v1" or not isinstance(matrix.get("runs"), list):
        raise ValueError("Research matrix requires matrix_version='research-matrix-v1' and runs")
    root = Path(artifact_root); runner = ResearchExperimentRunner(root)
    records: list[dict[str, Any]] = []
    for item in matrix["runs"]:
        name, config_path = item.get("name"), item.get("config")
        if not name or not config_path:
            raise ValueError("Every matrix run requires name and config")
        config = ResearchExperimentRunner.load_config((source.parent / config_path).resolve())
        overrides = dict(item.get("overrides") or {})
        for seed in item.get("seeds", [config.get("seed", 0)]):
            payload = {**config, **overrides, "seed": int(seed), "experiment_name": f"{name}-seed{seed}"}
            result = runner.run(payload, allow_dirty=allow_dirty)
            metrics = result.get("metrics") or {}
            records.append({"name": name, "seed": int(seed), "experiment_id": result["experiment_id"],
                            **{key: value for key, value in metrics.items() if isinstance(value, (int, float))}})
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(row["name"], []).append({key: value for key, value in row.items() if key not in {"name", "seed", "experiment_id"}})
    result = {"matrix_version": "research-matrix-v1", "matrix_sha256": sha256_file(source), "runs": records,
              "aggregates": {name: aggregate_repetitions(rows) for name, rows in grouped.items()}}
    target = root / "matrices" / f"{source.stem}.json"; target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Matrix artifact is immutable: {target}")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
