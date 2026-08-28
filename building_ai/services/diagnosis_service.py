"""Formal deterministic diagnosis orchestration; it never reruns semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from building_ai.core.diagnostics import DiagnosisConfig, run_rules
from building_ai.models import AnalysisResult, DiagnosticFinding
from building_ai.services.analytics_service import AnalyticsResult, AnalyticsService
from building_ai.services.equipment_service import EquipmentOrganization, EquipmentService
from building_ai.services.consistency_validation import AnalysisConsistencyValidation


@dataclass(slots=True)
class DiagnosisResult:
    findings: list[DiagnosticFinding] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    analytics: AnalyticsResult | None = None
    equipment: EquipmentOrganization | None = None
    consistency_validation: AnalysisConsistencyValidation | None = None


class DiagnosisService:
    def __init__(
        self, analytics: AnalyticsService | None = None,
        equipment: EquipmentService | None = None,
        config: DiagnosisConfig | None = None,
    ):
        self.analytics_service = analytics or AnalyticsService()
        self.equipment_service = equipment or EquipmentService()
        self.config = config or DiagnosisConfig()

    def diagnose_project(
        self, dataframe: pd.DataFrame, semantics: AnalysisResult, project_id: str,
        import_metadata: dict[str, Any] | None = None,
        progress_callback: Callable[[str, str, str | None, int, int], None] | None = None,
        organization: EquipmentOrganization | None = None,
        analytics_result: AnalyticsResult | None = None,
    ) -> DiagnosisResult:
        if progress_callback:
            progress_callback("equipment_grouping", "running", None, 0, 0)
        organization = organization or self.equipment_service.organize(project_id, semantics.semantic_results)
        if progress_callback:
            progress_callback("equipment_grouping", "completed", None, len(organization.heat_sources), len(organization.heat_sources))
            progress_callback("resolve_signals", "running", None, 0, 0)
            progress_callback("resolve_signals", "completed", None, len(organization.heat_sources), len(organization.heat_sources))
        analytics = analytics_result or self.analytics_service.analyze_project(
            dataframe, semantics, project_id, import_metadata, organization, progress_callback
        )
        if progress_callback:
            progress_callback("diagnostics", "running", None, 0, 0)
        evaluated = run_rules(analytics, self.config)
        unique: dict[tuple[str | None, str], DiagnosticFinding] = {}
        for finding in evaluated.findings:
            key = (finding.equipment_id, finding.finding_type)
            if key not in unique or finding.confidence > unique[key].confidence:
                unique[key] = finding
        if progress_callback:
            progress_callback("diagnostics", "completed", None, len(unique), len(unique))
        return DiagnosisResult(list(unique.values()), evaluated.skipped, analytics, organization)
