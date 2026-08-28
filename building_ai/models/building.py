from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(slots=True)
class Building:
    project_id: str
    name: str
    building_id: str = field(default_factory=lambda: str(uuid4()))
    latitude: float | None = None
    longitude: float | None = None
    timezone: str | None = None
