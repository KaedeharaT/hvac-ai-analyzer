import json
import sqlite3

from building_ai.storage import Database, ProjectStore


def test_v1_schema_is_migrated_without_losing_review(tmp_path):
    path = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript("""
    CREATE TABLE projects (project_id TEXT PRIMARY KEY, name TEXT, description TEXT,
      building_name TEXT, created_at TEXT, updated_at TEXT, payload_json TEXT);
    CREATE TABLE semantic_results (result_id TEXT PRIMARY KEY, project_id TEXT,
      raw_name TEXT, ai_label TEXT, status TEXT, payload_json TEXT,
      UNIQUE(project_id, raw_name));
    CREATE TABLE human_reviews (project_id TEXT, raw_name TEXT, human_label TEXT,
      human_note TEXT, verified_at TEXT, PRIMARY KEY(project_id, raw_name));
    """)
    project = {"name": "Old", "project_id": "p1", "description": "",
               "building_name": "", "created_at": "t", "updated_at": "t",
               "source_files": [], "latitude": None, "longitude": None,
               "timezone": None, "time_range": {}, "settings": {},
               "semantic_summary": {}, "analysis_summary": {}}
    conn.execute("INSERT INTO projects VALUES (?,?,?,?,?,?,?)",
                 ("p1", "Old", "", "", "t", "t", json.dumps(project)))
    payload = {"raw_name": "Power", "canonical_label": "heat_source_power",
               "result_id": "r1", "source_file": "old.xlsx", "sheet": "A"}
    conn.execute("INSERT INTO semantic_results VALUES (?,?,?,?,?,?)",
                 ("r1", "p1", "Power", "heat_source_power", "ACCEPT", json.dumps(payload)))
    conn.execute("INSERT INTO human_reviews VALUES (?,?,?,?,?)",
                 ("p1", "Power", "other", "legacy review", "t"))
    conn.commit(); conn.close()
    store = ProjectStore(Database(path))
    item = store.load_semantics("p1")[0]
    assert item.point_id
    assert item.canonical_label == "heat_source_power"
    assert item.human_label == "other"
    with sqlite3.connect(path) as migrated:
        assert migrated.execute("PRAGMA user_version").fetchone()[0] == 3
