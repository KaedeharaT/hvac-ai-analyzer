from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


# Latest paper baseline: strict target space used by the evaluated V3 pipeline.
STRICT_8_TAXONOMY = (
    "heat_source_supply_temp", "heat_source_return_temp", "heat_source_flow",
    "heat_source_power", "terminal_supply_air_temp", "terminal_return_air_temp",
    "terminal_power", "other",
)

# Retained solely to read projects created by the earlier product prototype.
LEGACY_13_TAXONOMY = (
    "heat_source_supply_temp", "heat_source_return_temp", "heat_source_flow",
    "heat_source_power", "heat_source_energy", "heat_source_capacity",
    "terminal_supply_air_temp", "terminal_return_air_temp",
    "terminal_air_volume", "terminal_power", "terminal_energy",
    "terminal_capacity", "other",
)

# UI review choices follow the current strict research protocol. Saved legacy
# results remain readable through SUPPORTED_TAXONOMY.
TAXONOMY = STRICT_8_TAXONOMY
SUPPORTED_TAXONOMY = tuple(dict.fromkeys((*STRICT_8_TAXONOMY, *LEGACY_13_TAXONOMY)))


class SemanticStatus(str, Enum):
    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    ABSTAIN = "ABSTAIN"


@dataclass(slots=True)
class SemanticResult:
    raw_name: str
    canonical_label: str
    result_id: str = field(default_factory=lambda: str(uuid4()))
    point_id: str | None = None
    canonical_name: str | None = None
    equipment_type: str | None = None
    equipment_id: str | None = None
    group_id: str | None = None
    relation_confidence: float | None = None
    physical_quantity: str | None = None
    signal_type: str | None = None  # instantaneous / cumulative / status / setpoint / measurement
    medium: str | None = None
    position: str | None = None
    unit: str | None = None
    confidence: float | None = None
    gate_status: str | None = None
    physical_validity: bool | None = None
    needs_review: bool = False
    abstained: bool = False
    suspicious: bool = False
    reason: str | None = None
    physics_warnings: list[str] = field(default_factory=list)
    per_label_scores: dict[str, Any] = field(default_factory=dict)
    debug_metadata: dict[str, Any] = field(default_factory=dict)
    human_verified: bool = False
    human_label: str | None = None
    human_note: str | None = None
    verified_at: str | None = None
    confirmed_label: str | None = None
    confirmed_equipment_id: str | None = None
    confirmed_at: str | None = None
    confirmation_source: str | None = None
    source_file: str | None = None
    sheet: str | None = None
    column: str | None = None
    analysis_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    model_provider: str | None = None
    model_name: str | None = None
    prompt_version: str | None = None
    algorithm_version: str = "research-c1-c8-v1"

    def __post_init__(self) -> None:
        if self.canonical_label not in SUPPORTED_TAXONOMY:
            raise ValueError(f"Unknown semantic label: {self.canonical_label}")

    @property
    def status(self) -> SemanticStatus:
        if self.abstained or str(self.gate_status).upper() == "ABSTAIN":
            return SemanticStatus.ABSTAIN
        if self.needs_review or self.suspicious:
            return SemanticStatus.REVIEW
        return SemanticStatus.ACCEPT

    @property
    def effective_label(self) -> str:
        if self.confirmed_label:
            return self.confirmed_label
        if self.human_verified and self.human_label:
            return self.human_label
        return self.canonical_label

    @property
    def effective_equipment_id(self) -> str | None:
        """Human-confirmed device ownership takes precedence over AI evidence."""
        return self.confirmed_equipment_id or self.equipment_id

    @property
    def review_status(self) -> str:
        return "CONFIRMED" if self.confirmed_label else self.status.value

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload
