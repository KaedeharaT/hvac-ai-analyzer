"""Evaluate real curated BuildingAI retrieval, including cross-language cases."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from building_ai.config import Settings
from building_ai.knowledge import KnowledgeService
from building_ai.knowledge_catalog import (
    CATALOG_DIR,
    load_materialized_facts,
    materialize_catalog,
    source_registry,
)
from building_ai.storage import Database

CASES = [
    ("rag-zh-low-delta", "Chinese", "冷冻水温差低应该先检查什么？", "buildingai_engineering_synthesis"),
    ("rag-en-low-delta", "English", "What should I inspect when chilled-water delta-T remains low?", "buildingai_engineering_synthesis"),
    ("rag-ja-low-delta", "Japanese", "冷水温度差が小さい場合、何を確認すべきですか？", "buildingai_engineering_synthesis"),
    ("rag-en-part-load", "English", "Why can chiller efficiency change at part load?", "us_energyplus"),
    ("rag-en-pump", "English", "What energy-saving directions apply when a chilled-water pump stays at high speed?", "us_doe_better_buildings"),
    ("rag-en-temperature", "English", "What should be considered before changing chilled-water supply temperature?", "us_energyplus"),
    ("rag-zh-cycling", "Chinese", "设备频繁启停应该检查什么？", "buildingai_engineering_synthesis"),
    ("rag-en-night-power", "English", "Why might power remain high overnight?", "us_doe_retuning"),
    ("rag-ja-retrofit", "Japanese", "既存建物の省エネ改修では何を評価すべきですか？", "jp_env_zeb_retrofit"),
    ("rag-zh-bems", "Chinese", "公共建筑能源管理中为什么要分项计量和实时监测？", "cn_mee_public_institutions"),
    ("rag-en-haystack", "English", "How should I describe a chilled water supply temperature BEMS point in Haystack?", "us_project_haystack"),
    ("rag-en-brick", "English", "What relationship should Brick equipment have with a sensor?", "us_brick"),
]


def _head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, help="Configured BuildingAI SQLite database by default.")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts" / "knowledge")
    arguments = parser.parse_args()
    database = Database(arguments.database or Settings.load().database_path); database.initialize()
    # Evaluate the versioned, on-disk corpus that the build command produces,
    # rather than an in-memory fixture.  Materialisation is deterministic and
    # makes the evaluation self-contained for a clean checkout.
    materialize_catalog(CATALOG_DIR)
    knowledge = KnowledgeService(database)
    stats = knowledge.ingest_catalog(source_registry(), load_materialized_facts(CATALOG_DIR))
    results = []
    for case_id, language, query, expected_source in CASES:
        retrieved = knowledge.search(query, top_k=3)
        source_ids = [item["metadata"].get("source_id") for item in retrieved]
        results.append({
            "case_id": case_id, "language": language, "query": query, "expected_source": expected_source,
            "top_1_hit": bool(source_ids and source_ids[0] == expected_source),
            "top_k_hit": expected_source in source_ids,
            "retrieved": [{"source_id": item["metadata"].get("source_id"), "citation": item["citation"], "score": item["score"]} for item in retrieved],
        })
    total = len(results)
    report = {
        "run_id": f"knowledge-rag-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "timestamp": datetime.now(timezone.utc).isoformat(), "commit": _head(), "case_count": total,
        "knowledge_stats": stats,
        "metrics": {
            "top_1_hit_rate": sum(item["top_1_hit"] for item in results) / total,
            "top_k_hit_rate": sum(item["top_k_hit"] for item in results) / total,
            "multilingual_top_k_hit_rate": sum(item["top_k_hit"] for item in results if item["case_id"].endswith("low-delta")) / 3,
        },
        "cases": results,
        "failed_cases": [item for item in results if not item["top_k_hit"]],
    }
    arguments.artifacts.mkdir(parents=True, exist_ok=True)
    (arguments.artifacts / "latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# BuildingAI knowledge RAG evaluation", "", f"- Commit: `{report['commit']}`", f"- Cases: {total}", f"- Top-1 hit rate: {report['metrics']['top_1_hit_rate']:.1%}", f"- Top-k hit rate: {report['metrics']['top_k_hit_rate']:.1%}", f"- Multilingual low-ΔT Top-k: {report['metrics']['multilingual_top_k_hit_rate']:.1%}", "", "## Cases", "", "| Case | Language | Top-1 | Top-k |", "| --- | --- | --- | --- |"]
    lines += [f"| {item['case_id']} | {item['language']} | {'PASS' if item['top_1_hit'] else 'FAIL'} | {'PASS' if item['top_k_hit'] else 'FAIL'} |" for item in results]
    (arguments.artifacts / "latest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    return 0 if not report["failed_cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
