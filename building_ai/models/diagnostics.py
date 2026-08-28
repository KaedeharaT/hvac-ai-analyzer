"""Traceable domain objects for deterministic BEMS diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class AnalysisEvidence:
    point_ids: list[str]
    equipment_id: str | None
    start_time: str | None
    end_time: str | None
    metric_name: str
    metric_value: Any
    unit: str | None
    threshold: Any = None
    calculation_method: str = ""
    source_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DiagnosticFinding:
    finding_id: str
    equipment_id: str | None
    finding_type: str
    severity: str
    title: str
    description: str
    evidence: list[AnalysisEvidence] = field(default_factory=list)
    confidence: float = 0.0
    affected_period: dict[str, str | None] = field(default_factory=dict)
    source_metrics: list[str] = field(default_factory=list)
    status: str = "open"
    valid_sample_count: int = 0
    occurrence_count: int = 0
    duration_hours: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EnergySavingOpportunity:
    opportunity_id: str
    related_finding_ids: list[str]
    equipment_id: str | None
    category: str
    title: str
    recommendation: str
    expected_impact: str
    implementation_difficulty: str
    priority: str
    evidence: list[AnalysisEvidence] = field(default_factory=list)
    confidence: float = 0.0
    llm_explanation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
