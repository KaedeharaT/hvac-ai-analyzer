from building_ai.knowledge import KnowledgeService
import json

from building_ai.knowledge_catalog import curated_facts, materialize_catalog, source_registry
from building_ai.storage import Database


def test_curated_catalog_is_persistent_idempotent_and_multilingual(tmp_path):
    database = Database(tmp_path / "knowledge.sqlite3"); database.initialize()
    knowledge = KnowledgeService(database)
    first = knowledge.ingest_catalog(source_registry(), curated_facts())
    second = knowledge.ingest_catalog(source_registry(), curated_facts())

    assert first["source_count"] >= 18
    assert first["chunk_count"] >= 100
    assert first["sources_by_country"]["China"] >= 5
    assert first["sources_by_country"]["US"] >= 8
    assert first["sources_by_country"]["Japan"] >= 5
    assert first["chunk_count"] == second["chunk_count"]

    for query in (
        "冷冻水温差低应该先检查什么？",
        "What should I inspect when chilled-water delta-T remains low?",
        "冷水温度差が小さい場合、何を確認すべきですか？",
    ):
        results = knowledge.search(query, top_k=3)
        assert results
        assert any(item["metadata"]["source_id"] == "buildingai_engineering_synthesis" for item in results)
        assert all(item["citation"] and item["metadata"]["official_url"] for item in results)


def test_structured_semantic_sources_are_retrievable(tmp_path):
    database = Database(tmp_path / "knowledge.sqlite3"); database.initialize()
    knowledge = KnowledgeService(database); knowledge.ingest_catalog(source_registry(), curated_facts())
    results = knowledge.search("What relationship should a Brick equipment have with a sensor?", top_k=3)
    assert results
    assert any(item["metadata"]["source_id"] == "us_brick" for item in results)


def test_catalog_materializes_curated_chunks_metadata_and_keyword_index(tmp_path):
    root = tmp_path / "knowledge"
    manifest = materialize_catalog(root)
    chunks = root / "chunks" / "knowledge_chunks.jsonl"
    assert manifest["chunk_count"] >= 150
    assert chunks.exists() and len(chunks.read_text(encoding="utf-8").splitlines()) == manifest["chunk_count"]
    assert (root / "curated" / "china" / "knowledge.jsonl").exists()
    assert (root / "curated" / "us" / "knowledge.jsonl").exists()
    assert (root / "curated" / "japan" / "knowledge.jsonl").exists()
    assert (root / "metadata" / "concept_aliases.json").exists()
    index = json.loads((root / "index" / "keyword_cjk_index.json").read_text(encoding="utf-8"))
    assert "chilled_water_pump" in index
