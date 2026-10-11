"""Project-scoped model snapshots and explicit equipment-reference evidence."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from building_ai.vision.model_geometry import MAX_BYTES, read_osm
from building_ai.services.equipment_service import EquipmentService
from building_ai.storage import ProjectStore


def _write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class ModelViewerService:
    def __init__(self, database, root: str | Path):
        self.database, self.root = database, Path(root).resolve()

    def _directory(self, project_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", project_id):
            raise KeyError("Invalid project")
        with self.database.connect() as connection:
            if not connection.execute("SELECT 1 FROM projects WHERE project_id=?", (project_id,)).fetchone():
                raise KeyError("Unknown project")
        path = (self.root / project_id / "models").resolve()
        path.relative_to(self.root)
        return path

    def _model_dir(self, project_id: str, model_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{64}", model_id):
            raise KeyError("Invalid model ID")
        return self._directory(project_id) / model_id

    def import_model(self, project_id: str, source: str | Path) -> dict:
        source = Path(source)
        if source.suffix.lower() != ".osm" or source.stat().st_size > MAX_BYTES:
            raise ValueError("Select an OSM file within the 20 MiB limit")
        data = source.read_bytes()
        geometry = read_osm(data)
        directory = self._model_dir(project_id, geometry["model_hash"])
        if (directory / "manifest.json").exists():
            return self.load(project_id, geometry["model_hash"])
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "source.osm").write_bytes(data)
        _write(directory / "manifest.json", {"model_id": geometry["model_hash"], "project_id": project_id,
            "source_name": source.name, "imported_at": datetime.now(timezone.utc).isoformat()})
        _write(directory / "associations.json", {"events": []})
        return self.load(project_id, geometry["model_hash"])

    def list_models(self, project_id: str) -> list[dict]:
        return [json.loads(p.read_text(encoding="utf-8"))
                for p in sorted(self._directory(project_id).glob("*/manifest.json"))]

    def load(self, project_id: str, model_id: str) -> dict:
        directory = self._model_dir(project_id, model_id)
        if not (directory / "manifest.json").exists():
            raise KeyError("Model is not part of this project")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["project_id"] != project_id or manifest["model_id"] != model_id:
            raise ValueError("Model provenance mismatch")
        source = directory / "source.osm"
        if source.stat().st_size > MAX_BYTES:
            raise ValueError("Model snapshot exceeds size limit")
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != model_id:
            raise ValueError("Model snapshot changed; import and review it again")
        return {**read_osm(data), **manifest}

    def associations(self, project_id: str, model_id: str) -> list[dict]:
        self.load(project_id, model_id)
        path = self._model_dir(project_id, model_id) / "associations.json"
        return json.loads(path.read_text(encoding="utf-8"))["events"]

    def associate(self, project_id: str, model_id: str, object_id: str,
                  equipment_id: str, reviewer: str, note: str) -> dict:
        model = self.load(project_id, model_id)
        if not any(e["object_id"] == object_id for e in model["elements"]):
            raise KeyError("Unknown model object")
        semantics = ProjectStore(self.database).load_semantics(project_id)
        equipment = EquipmentService().organize(project_id, semantics).equipment
        if not any(e.equipment_id == equipment_id for e in equipment):
            raise ValueError("Equipment is not part of this project's current semantic context")
        if not reviewer.strip() or not note.strip():
            raise ValueError("Reviewer and association evidence are required")
        event = {"project_id": project_id, "model_hash": model_id, "object_id": object_id,
            "equipment_id": equipment_id, "association_kind": "reference_only_not_equipment_position",
            "review_status": "human_confirmed", "reviewer": reviewer.strip(), "note": note.strip(),
            "confirmed_at": datetime.now(timezone.utc).isoformat(),
            "coordinate_system": model["coordinate_system"], "unit": model["unit"],
            "transform_method": model["transform_method"], "drawing_hash": None}
        path = self._model_dir(project_id, model_id) / "associations.json"
        data = json.loads(path.read_text(encoding="utf-8")); data["events"].append(event)
        _write(path, data)
        return event

    def export_provenance(self, project_id: str, model_id: str, destination: str | Path) -> None:
        model = self.load(project_id, model_id)
        payload = {key: model[key] for key in ("project_id", "model_id", "model_hash", "source_name",
            "source_format", "source_version", "adapter_version", "coordinate_system", "unit", "up_axis",
            "transform_method", "unit_reference", "engineering_validation", "imported_at", "boundary",
            "omitted_geometry")}
        payload["drawing_hash"] = None  # No image-to-model registration exists.
        payload["association_events"] = self.associations(project_id, model_id)
        payload["objects"] = [{k: e[k] for k in ("object_id", "name", "category", "space_id")}
                              for e in model["elements"]]
        _write(Path(destination), payload)
