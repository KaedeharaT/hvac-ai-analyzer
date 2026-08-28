from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from building_ai.models import make_point_id


SCHEMA_VERSION = 3

BASE_SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL,
    building_name TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS points (
    point_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, source_file TEXT,
    sheet TEXT, raw_name TEXT NOT NULL, payload_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(project_id, source_file, sheet, raw_name),
    FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS semantic_results (
    result_id TEXT PRIMARY KEY, point_id TEXT NOT NULL UNIQUE,
    project_id TEXT NOT NULL, ai_label TEXT NOT NULL, status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY(point_id) REFERENCES points(point_id) ON DELETE CASCADE,
    FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS human_reviews (
    point_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, human_label TEXT,
    human_equipment_id TEXT, human_note TEXT, verified_at TEXT NOT NULL,
    FOREIGN KEY(point_id) REFERENCES points(point_id) ON DELETE CASCADE,
    FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS import_metadata (
    project_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL,
    FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS equipment (
    equipment_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, payload_json TEXT NOT NULL,
    FOREIGN KEY(project_id) REFERENCES projects(project_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS agent_sessions (
    session_id TEXT PRIMARY KEY, project_id TEXT, payload_json TEXT NOT NULL
);
"""


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if version == 0 and self._is_v1_database(conn):
                self._migrate_v1_to_v2(conn)
            conn.executescript(BASE_SCHEMA)
            self._migrate_v2_to_v3(conn)
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    @staticmethod
    def _migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(human_reviews)").fetchall()}
        if columns and "human_equipment_id" not in columns:
            conn.execute("ALTER TABLE human_reviews ADD COLUMN human_equipment_id TEXT")

    @staticmethod
    def _is_v1_database(conn: sqlite3.Connection) -> bool:
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(semantic_results)").fetchall()
        }
        return bool(columns) and "point_id" not in columns

    def _migrate_v1_to_v2(self, conn: sqlite3.Connection) -> None:
        """Preserve development V1 records while replacing raw-name identity."""
        semantics = conn.execute("SELECT * FROM semantic_results").fetchall()
        reviews = {
            (row["project_id"], row["raw_name"]): row
            for row in conn.execute("SELECT * FROM human_reviews").fetchall()
        }
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("ALTER TABLE semantic_results RENAME TO semantic_results_v1")
        conn.execute("ALTER TABLE human_reviews RENAME TO human_reviews_v1")
        conn.executescript(BASE_SCHEMA)
        for row in semantics:
            payload = json.loads(row["payload_json"])
            source_file = payload.get("source_file")
            sheet = payload.get("sheet")
            point_id = payload.get("point_id") or make_point_id(
                row["project_id"], source_file, sheet, row["raw_name"]
            )
            payload["point_id"] = point_id
            conn.execute(
                "INSERT OR IGNORE INTO points VALUES (?, ?, ?, ?, ?, ?)",
                (point_id, row["project_id"], source_file, sheet, row["raw_name"], "{}"),
            )
            conn.execute(
                "INSERT INTO semantic_results VALUES (?, ?, ?, ?, ?, ?)",
                (row["result_id"], point_id, row["project_id"], row["ai_label"],
                 row["status"], json.dumps(payload, ensure_ascii=False)),
            )
            review = reviews.get((row["project_id"], row["raw_name"]))
            if review:
                conn.execute(
                    "INSERT INTO human_reviews (point_id, project_id, human_label, human_note, verified_at) VALUES (?, ?, ?, ?, ?)",
                    (point_id, row["project_id"], review["human_label"], review["human_note"], review["verified_at"]),
                )
        conn.execute("DROP TABLE semantic_results_v1")
        conn.execute("DROP TABLE human_reviews_v1")
        conn.execute("PRAGMA foreign_keys=ON")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
