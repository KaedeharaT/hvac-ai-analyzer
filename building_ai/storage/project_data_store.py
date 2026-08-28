"""Managed project import copies, kept separate from source-file locations."""
from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path


class DuplicateImportError(ValueError):
    """Raised when an identical source file already belongs to a project."""


class ProjectDataStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def project_dir(self, project_id: str) -> Path:
        target = (self.root / project_id).resolve()
        if target.parent != self.root.resolve():
            raise ValueError("Invalid project identifier")
        return target

    @staticmethod
    def file_hash(source: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(source).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def copy_import(self, project_id: str, source: str | Path, existing: list[dict], *, allow_duplicate: bool = False) -> dict:
        source = Path(source)
        digest = self.file_hash(source)
        if any(item.get("file_hash") == digest for item in existing) and not allow_duplicate:
            raise DuplicateImportError("This file has already been imported into the current project.")
        imports = self.project_dir(project_id) / "imports"
        imports.mkdir(parents=True, exist_ok=True)
        target = imports / f"{digest[:12]}_{source.name}"
        if target.exists():
            target = imports / f"{digest[:12]}_{len(existing) + 1}_{source.name}"
        shutil.copy2(source, target)
        return {
            "original_filename": source.name,
            "source_path": str(source.resolve()),  # audit-only; runtime never reads this path
            "managed_path": str(target),
            "file_hash": digest,
            "file_size": source.stat().st_size,
            "imported_at": datetime.now(timezone.utc).isoformat(),
            "source_type": source.suffix.lower().lstrip("."),
            "duplicate_of_existing": any(item.get("file_hash") == digest for item in existing),
        }

    def clear(self, project_id: str) -> None:
        target = self.project_dir(project_id)
        if target.exists():
            shutil.rmtree(target)
