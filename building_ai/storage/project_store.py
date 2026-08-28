from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

from building_ai.models import Project, SemanticResult, make_point_id

from .database import Database


class ProjectStore:
    def __init__(self, database: Database):
        self.database = database
        self.database.initialize()

    def save(self, project: Project) -> Project:
        project.updated_at = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(asdict(project), ensure_ascii=False)
        with self.database.connect() as conn:
            conn.execute(
                """INSERT INTO projects VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET name=excluded.name,
                description=excluded.description, building_name=excluded.building_name,
                updated_at=excluded.updated_at, payload_json=excluded.payload_json""",
                (project.project_id, project.name, project.description,
                 project.building_name, project.created_at, project.updated_at, payload),
            )
        return project

    def create(self, name: str, building_name: str = "", description: str = "") -> Project:
        return self.save(Project(name=name, building_name=building_name, description=description))

    def get(self, project_id: str) -> Project | None:
        with self.database.connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM projects WHERE project_id=?", (project_id,)
            ).fetchone()
        return Project(**json.loads(row[0])) if row else None

    def list(self) -> list[Project]:
        with self.database.connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM projects ORDER BY updated_at DESC"
            ).fetchall()
        return [Project(**json.loads(row[0])) for row in rows]

    def rename(self, project_id: str, name: str) -> Project:
        project = self.get(project_id)
        if project is None:
            raise KeyError(project_id)
        project.name = name
        return self.save(project)

    def delete(self, project_id: str) -> None:
        with self.database.connect() as conn:
            conn.execute("DELETE FROM projects WHERE project_id=?", (project_id,))

    def save_semantics(self, project_id: str, results: list[SemanticResult]) -> None:
        with self.database.connect() as conn:
            for item in results:
                item.point_id = item.point_id or make_point_id(
                    project_id, item.source_file, item.sheet, item.raw_name
                )
                payload = json.dumps(item.to_dict(), ensure_ascii=False, default=str)
                conn.execute(
                    """INSERT INTO points
                    (point_id, project_id, source_file, sheet, raw_name, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(point_id) DO UPDATE SET payload_json=excluded.payload_json""",
                    (item.point_id, project_id, item.source_file, item.sheet,
                     item.raw_name, "{}"),
                )
                conn.execute(
                    """INSERT INTO semantic_results
                    (result_id, point_id, project_id, ai_label, status, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(point_id) DO UPDATE SET
                    result_id=excluded.result_id, ai_label=excluded.ai_label,
                    status=excluded.status, payload_json=excluded.payload_json""",
                    (item.result_id, item.point_id, project_id, item.canonical_label,
                     item.status.value, payload),
                )

    def load_semantics(self, project_id: str) -> list[SemanticResult]:
        with self.database.connect() as conn:
            rows = conn.execute(
                """SELECT s.payload_json, r.human_label, r.human_equipment_id, r.human_note, r.verified_at
                FROM semantic_results s LEFT JOIN human_reviews r
                ON s.point_id=r.point_id
                WHERE s.project_id=? ORDER BY s.rowid""", (project_id,)
            ).fetchall()
        items: list[SemanticResult] = []
        for row in rows:
            payload = json.loads(row[0])
            payload.pop("status", None)
            if row[4]:
                payload.update(human_verified=True, human_label=row[1],
                               confirmed_equipment_id=row[2], human_note=row[3], verified_at=row[4],
                               confirmed_label=row[1], confirmed_at=row[4],
                               confirmation_source="human_confirmed")
            items.append(SemanticResult(**payload))
        return items

    def save_review(
        self, project_id: str, point_id: str, human_label: str | None, note: str = "", human_equipment_id: str | None = None,
    ) -> None:
        verified_at = datetime.now(timezone.utc).isoformat()
        with self.database.connect() as conn:
            point = conn.execute(
                "SELECT 1 FROM points WHERE point_id=? AND project_id=?",
                (point_id, project_id),
            ).fetchone()
            if not point:
                raise KeyError(f"Unknown point_id for project: {point_id}")
            conn.execute(
                """INSERT INTO human_reviews (point_id, project_id, human_label, human_equipment_id, human_note, verified_at) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(point_id) DO UPDATE SET
                human_label=excluded.human_label, human_equipment_id=excluded.human_equipment_id, human_note=excluded.human_note,
                verified_at=excluded.verified_at""",
                (point_id, project_id, human_label, human_equipment_id, note, verified_at),
            )

    def clear_runtime_results(self, project_id: str, *, clear_import_metadata: bool = False) -> None:
        """Invalidate project-local semantics/analysis dependencies, never global knowledge."""
        with self.database.connect() as conn:
            conn.execute("DELETE FROM points WHERE project_id=?", (project_id,))
            if clear_import_metadata:
                conn.execute("DELETE FROM import_metadata WHERE project_id=?", (project_id,))

    def get_semantic(self, project_id: str, point_id: str) -> SemanticResult | None:
        return next(
            (item for item in self.load_semantics(project_id) if item.point_id == point_id),
            None,
        )

    def save_import_metadata(self, project_id: str, metadata: dict) -> None:
        with self.database.connect() as conn:
            conn.execute(
                """INSERT INTO import_metadata VALUES (?, ?)
                ON CONFLICT(project_id) DO UPDATE SET payload_json=excluded.payload_json""",
                (project_id, json.dumps(metadata, default=str)),
            )

    def get_import_metadata(self, project_id: str) -> dict:
        with self.database.connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM import_metadata WHERE project_id=?", (project_id,)
            ).fetchone()
        return json.loads(row[0]) if row else {}
