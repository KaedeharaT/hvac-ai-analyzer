from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4


class EquipmentType(str, Enum):
    HEAT_SOURCE = "Heat Source"
    CHILLER = "Chiller"
    HEAT_PUMP = "Heat Pump"
    AHU = "AHU"
    FCU = "FCU"
    PUMP = "Pump"
    FAN = "Fan"
    UNKNOWN = "Unknown"


@dataclass(slots=True)
class Equipment:
    project_id: str
    name: str
    equipment_type: EquipmentType = EquipmentType.UNKNOWN
    equipment_id: str = field(default_factory=lambda: str(uuid4()))
    building_id: str | None = None
    confidence: float | None = None
