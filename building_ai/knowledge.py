"""Persistent, source-aware local retrieval for BuildingAI knowledge."""
from __future__ import annotations

import json
import re
import uuid
from collections import Counter


def _search_terms(text: str) -> set[str]:
    """Keyword plus CJK n-gram terms; always available without an embedding model."""
    normalized = text.casefold()
    terms = set(re.findall(r"[\w-]+", normalized))
    for span in re.findall(r"[\u4e00-\u9fffぁ-んァ-ンー]+", normalized):
        terms.add(span)
        terms.update(span[index:index + 2] for index in range(max(0, len(span) - 1)))
        terms.update(span[index:index + 3] for index in range(max(0, len(span) - 2)))
    return {term for term in terms if len(term) > 1}


class KnowledgeService:
    """SQLite-backed retrieval with stable catalog IDs and transparent metadata."""

    def __init__(self, database):
        self.database = database
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.database.connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS knowledge_chunks (chunk_id TEXT PRIMARY KEY,title TEXT,source TEXT,section TEXT,text TEXT,metadata TEXT)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS knowledge_sources (source_id TEXT PRIMARY KEY,payload_json TEXT NOT NULL)"
            )

    def ingest(self, title, source, text, section="General", metadata=None, chunk_size=800):
        """Ingest a caller-supplied source while retaining provenance per chunk."""
        if not title.strip() or not source.strip() or not text.strip():
            raise ValueError("title, source and text are required")
        if chunk_size < 40:
            raise ValueError("chunk_size must be at least 40")
        payload = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)
        words = re.findall(r"\S+(?:\s+|$)", text.strip())
        chunks, current = [], ""
        for word in words:
            if current and len(current) + len(word) > chunk_size:
                chunks.append(current.strip()); current = ""
            current += word
        if current.strip():
            chunks.append(current.strip())
        identifiers = []
        with self.database.connect() as connection:
            for index, chunk in enumerate(chunks, 1):
                chunk_id = str(uuid.uuid4())
                connection.execute("INSERT INTO knowledge_chunks VALUES (?,?,?,?,?,?)", (chunk_id, title, source, f"{section} ({index}/{len(chunks)})", chunk, payload))
                identifiers.append(chunk_id)
        return identifiers

    def ingest_catalog(self, registry: list[dict], facts: list[dict]) -> dict:
        """Idempotently load curated public summaries into the formal storage."""
        registry_by_id = {item["source_id"]: item for item in registry}
        with self.database.connect() as connection:
            # Catalog facts are replaceable release data.  Leave any
            # user-supplied chunks intact while removing obsolete catalog IDs.
            connection.execute("DELETE FROM knowledge_chunks WHERE chunk_id LIKE 'catalog:%'")
            for source in registry:
                connection.execute(
                    "INSERT OR REPLACE INTO knowledge_sources VALUES (?,?)",
                    (source["source_id"], json.dumps(source, ensure_ascii=False, sort_keys=True)),
                )
            for fact in facts:
                source = registry_by_id[fact["source_id"]]
                metadata = {
                    "record_id": fact["record_id"], "source_id": fact["source_id"], "country": fact["country"],
                    "language": fact["language"], "knowledge_category": fact["knowledge_category"],
                    "equipment_type": fact["equipment_type"], "concepts": fact["concepts"],
                    "organization": source["organization"], "official_url": source["official_url"],
                    "license_note": source["license_or_usage_note"], "content_strategy": source["content_strategy"],
                    "citation": f"{source['organization']} — {source['title']} — {fact['section']}",
                }
                connection.execute(
                    "INSERT OR REPLACE INTO knowledge_chunks VALUES (?,?,?,?,?,?)",
                    (f"catalog:{fact['record_id']}", fact["title"], source["official_url"], fact["section"], fact["text"],
                     json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
                )
        return self.stats()

    @staticmethod
    def _query_concepts(query: str) -> set[str]:
        # Controlled vocabulary bridges languages; it is catalog data, not
        # per-question router logic.
        from building_ai.knowledge_catalog import curated_facts
        text = query.casefold()
        matches: dict[str, int] = {}
        for fact in curated_facts():
            if fact["equipment_type"] != "concept_dictionary":
                continue
            concept = fact["title"]
            aliases = re.findall(r"Aliases:\s*(.+?)\.", fact["text"], flags=re.IGNORECASE)
            values = [concept, *(aliases[0].split(", ") if aliases else [])]
            hit_lengths = [len(value) for value in values if value.casefold() in text]
            if hit_lengths:
                matches[concept] = max(hit_lengths)
        # A specific multilingual alias (for example "low delta-T") should
        # not be diluted by a shorter nested alias such as "water temperature".
        if not matches:
            return set()
        longest = max(matches.values())
        return {concept for concept, length in matches.items() if length >= longest * 0.8}

    def search(self, query, top_k=3):
        query_terms = _search_terms(query)
        query_concepts = self._query_concepts(query)
        with self.database.connect() as connection:
            rows = connection.execute("SELECT * FROM knowledge_chunks").fetchall()
        scored = []
        for row in rows:
            metadata = json.loads(row["metadata"])
            # Concept-dictionary rows power multilingual expansion but are not
            # explanatory sources shown to an end user.
            if metadata.get("equipment_type") == "concept_dictionary":
                continue
            document_terms = _search_terms(" ".join((row["title"], row["section"], row["text"], " ".join(metadata.get("concepts", [])))))
            lexical = len(query_terms & document_terms)
            concept_overlap = len(query_concepts & set(metadata.get("concepts", [])))
            score = lexical + concept_overlap * 12
            if score <= 0:
                continue
            scored.append({
                "chunk_id": row["chunk_id"], "title": row["title"], "source": row["source"], "section": row["section"],
                "text": row["text"], "metadata": metadata, "citation": metadata.get("citation") or f"{row['title']} — {row['section']} ({row['source']})",
                "score": score,
            })
        return sorted(scored, key=lambda item: (-item["score"], item["title"]))[:top_k]

    def stats(self) -> dict:
        with self.database.connect() as connection:
            source_rows = connection.execute("SELECT payload_json FROM knowledge_sources").fetchall()
            chunk_rows = connection.execute("SELECT metadata FROM knowledge_chunks").fetchall()
        sources = [json.loads(row["payload_json"]) for row in source_rows]
        chunks = [json.loads(row["metadata"]) for row in chunk_rows]
        return {
            "source_count": len(sources), "chunk_count": len(chunks),
            "sources_by_country": dict(Counter(source.get("country", "Unknown") for source in sources)),
            "chunks_by_country": dict(Counter(chunk.get("country", "Unknown") for chunk in chunks)),
            "chunks_by_language": dict(Counter(chunk.get("language", "Unknown") for chunk in chunks)),
            "chunks_by_category": dict(Counter(chunk.get("knowledge_category", "Unknown") for chunk in chunks)),
        }
