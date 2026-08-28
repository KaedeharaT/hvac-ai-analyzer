from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class Project:
    name: str
    project_id: str = field(default_factory=lambda: str(uuid4()))
    description: str = ""
    building_name: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    source_files: list[str] = field(default_factory=list)
    data_revision: int = 0
    data_status: str = "empty"
    latitude: float | None = None
    longitude: float | None = None
    timezone: str | None = None
    time_range: dict[str, str | None] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    semantic_summary: dict[str, Any] = field(default_factory=dict)
    analysis_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
