"""Build BuildingAI's curated, source-attributed local knowledge database."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from building_ai.config import Settings
from building_ai.knowledge import KnowledgeService
from building_ai.knowledge_catalog import CATALOG_DIR, load_materialized_facts, materialize_catalog, source_registry
from building_ai.storage import Database


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, help="Optional SQLite database path; defaults to the configured BuildingAI database.")
    arguments = parser.parse_args()
    path = arguments.database or Settings.load().database_path
    database = Database(path); database.initialize()
    manifest = materialize_catalog(CATALOG_DIR)
    stats = KnowledgeService(database).ingest_catalog(source_registry(), load_materialized_facts(CATALOG_DIR))
    print(json.dumps({
        "knowledge_source_registry": str(CATALOG_DIR / "source_registry.json"),
        "knowledge_curated_content": str(CATALOG_DIR / "curated"),
        "knowledge_chunks": str(CATALOG_DIR / "chunks" / "knowledge_chunks.jsonl"),
        "knowledge_index": str(CATALOG_DIR / "index" / "keyword_cjk_index.json"),
        "knowledge_database": str(path), "materialized": manifest, "persistent": stats,
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
