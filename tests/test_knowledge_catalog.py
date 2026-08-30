from building_ai.knowledge import KnowledgeService
from building_ai.knowledge_catalog import curated_facts, source_registry
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
