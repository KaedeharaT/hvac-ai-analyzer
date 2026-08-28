"""Application-local confirmed semantic dataset, separate from paper GT."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from building_ai.core.preprocessing import normalize_header
from building_ai.models import SemanticResult


def normalized_key(name: str) -> str:
    return normalize_header(name).casefold()


@dataclass(frozen=True, slots=True)
class ConfirmedMapping:
    normalized_name: str
    confirmed_label: str
    source_project_id: str
    source_file: str | None
    source_sheet: str | None
    unit: str | None
    equipment_type: str | None
    equipment_id: str | None
    group_id: str | None
    physical_quantity: str | None
    note: str
    confirmed_at: str


class ConfirmedMappingStore:
    """Auditable exact-match reuse only; fuzzy names never auto-accept."""
    def __init__(self, path: str | Path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS confirmed_semantic_mappings (
                mapping_id INTEGER PRIMARY KEY AUTOINCREMENT, normalized_name TEXT NOT NULL,
                original_name TEXT NOT NULL, confirmed_label TEXT NOT NULL, source TEXT NOT NULL,
                source_project_id TEXT NOT NULL, source_file TEXT, source_sheet TEXT,
                unit TEXT, equipment_type TEXT, equipment_id TEXT, group_id TEXT, physical_quantity TEXT, note TEXT NOT NULL,
                confirmed_at TEXT NOT NULL)""")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(confirmed_semantic_mappings)").fetchall()}
            if "equipment_id" not in columns:
                conn.execute("ALTER TABLE confirmed_semantic_mappings ADD COLUMN equipment_id TEXT")
            if "group_id" not in columns:
                conn.execute("ALTER TABLE confirmed_semantic_mappings ADD COLUMN group_id TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_confirmed_mapping_name ON confirmed_semantic_mappings(normalized_name)")

    def _connect(self):
        return sqlite3.connect(self.path)

    def save(self, project_id: str, item: SemanticResult, note: str = "") -> None:
        label = item.effective_label
        if not label: raise ValueError("A confirmed mapping requires a label")
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("""INSERT INTO confirmed_semantic_mappings
                (normalized_name, original_name, confirmed_label, source, source_project_id, source_file, source_sheet, unit, equipment_type, equipment_id, group_id, physical_quantity, note, confirmed_at)
                VALUES (?, ?, ?, 'human_confirmed', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (normalized_key(item.raw_name), item.raw_name, label, project_id, item.source_file, item.sheet, item.unit, item.equipment_type, item.effective_equipment_id, item.group_id, item.physical_quantity, note, now))

    def exact(self, name: str, unit: str | None = None) -> ConfirmedMapping | None:
        with self._connect() as conn:
            rows = conn.execute("""SELECT normalized_name, confirmed_label, source_project_id, source_file, source_sheet, unit, equipment_type, equipment_id, group_id, physical_quantity, note, confirmed_at
                FROM confirmed_semantic_mappings WHERE normalized_name=? ORDER BY confirmed_at DESC""", (normalized_key(name),)).fetchall()
        if not rows: return None
        # Conflicting historic labels are evidence for review, never a silent overwrite.
        if len({row[1] for row in rows}) != 1: return None
        row = rows[0]
        if unit and row[5] and str(unit).casefold() != str(row[5]).casefold(): return None
        return ConfirmedMapping(*row)
