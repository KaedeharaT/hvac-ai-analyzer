"""One product-facing analysis entry point.

Research source: paper_research/run_bems_semantic.py (separation of inference
from evaluation) and V3/V4 structured semantic evidence.  This orchestration
contains no experiment or ground-truth dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from building_ai.config import Settings
from building_ai.models import AnalysisResult
from .analytics_service import AnalyticsResult, AnalyticsService
from .diagnosis_service import DiagnosisResult, DiagnosisService
from .equipment_service import EquipmentOrganization, EquipmentService
from .energy_analysis_service import EnergyAnalysisResult, EnergyAnalysisService
from .semantic_service import SemanticService


@dataclass(slots=True)
class ProjectAnalysisRun:
    semantics: AnalysisResult
    equipment: EquipmentOrganization
    analytics: AnalyticsResult
    diagnosis: DiagnosisResult
    energy: EnergyAnalysisResult


class BuildingAnalysisPipeline:
    """Run semantic mapping, evidence-backed grouping, KPI and diagnosis safely."""
    def __init__(self, settings: Settings | None = None):
        self.semantics = SemanticService(settings)
        self.equipment = EquipmentService()
        self.analytics = AnalyticsService()
        self.diagnosis = DiagnosisService(self.analytics, self.equipment)
        self.energy = EnergyAnalysisService()

    def run(self, dataframe: pd.DataFrame, project_id: str = "<unbound-project>", *, source_file: str | None = None, sheet: str | None = None, import_metadata: dict[str, Any] | None = None, enhanced: bool = False) -> ProjectAnalysisRun:
        semantics = self.semantics.analyze_dataframe(dataframe, source_file=source_file, sheet=sheet, project_id=project_id, backend="enhanced" if enhanced else "offline")
        equipment = self.equipment.organize(project_id, semantics.semantic_results)
        analytics = self.analytics.analyze_project(dataframe, semantics, project_id, import_metadata, equipment)
        diagnosis = self.diagnosis.diagnose_project(dataframe, semantics, project_id, import_metadata, organization=equipment, analytics_result=analytics)
        energy = self.energy.analyze(dataframe, semantics, project_id, import_metadata, equipment, analytics)
        return ProjectAnalysisRun(semantics, equipment, analytics, diagnosis, energy)
