from __future__ import annotations

import pandas as pd

from building_ai.config import Settings
from building_ai.agent_controller import AgentController
from building_ai.core.equipment_identity import normalize_equipment_id
from building_ai.llm import LLMManager
from building_ai.models import AnalysisResult, Project
from building_ai.services.agent_service import AgentService
from building_ai.services.analytics_service import AnalyticsService
from building_ai.services.diagnosis_service import DiagnosisResult, DiagnosisService
from building_ai.services.equipment_service import EquipmentOrganization, EquipmentService
from building_ai.services.import_service import ImportResult, ImportService
from building_ai.services.semantic_service import PIPELINE_VERSION, SemanticService
from building_ai.services.opportunity_service import OpportunityService
from building_ai.services.consistency_validation import validate_analysis
from building_ai.services.energy_analysis_service import EnergyAnalysisResult, EnergyAnalysisService
from building_ai.services.interpretation_service import InterpretationService
from building_ai.services.drawing_service import DrawingService
from building_ai.storage import (
    ConfirmedMappingStore, Database, DuplicateImportError, ProjectDataStore,
    ProjectStore, TimeseriesStore,
)


class ApplicationContext:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.load()
        self.database = Database(self.settings.database_path)
        self.projects = ProjectStore(self.database)
        self.timeseries = TimeseriesStore(self.settings.timeseries_dir)
        self.project_data = ProjectDataStore(self.settings.timeseries_dir)
        self.drawings = DrawingService(self.database, self.settings.timeseries_dir)
        self.confirmed_mappings = ConfirmedMappingStore(self.settings.confirmed_dataset_path)
        self.importer = ImportService()
        self.semantics = SemanticService(self.settings)
        self.llm_manager = LLMManager(self.settings)
        self.equipment_service = EquipmentService()
        self.analytics = AnalyticsService()
        self.energy_analysis = EnergyAnalysisService()
        self.interpretation = InterpretationService()
        self.diagnosis = DiagnosisService(self.analytics, self.equipment_service)
        self.opportunities_service = OpportunityService(self.llm_manager)
        self.agent = AgentService(self.projects, self.timeseries, lambda: self.diagnosis_result, lambda: self.equipment_organization, lambda: self.energy_analysis_result, self.agent_project_state, self.drawings)
        self.current_project: Project | None = None
        self.dataframe: pd.DataFrame | None = None
        self.semantic_result: AnalysisResult | None = None
        self.import_metadata: dict = {}
        self.data_notice: str | None = None
        self.equipment = []
        self.equipment_organization: EquipmentOrganization | None = None
        self.diagnosis_result: DiagnosisResult | None = None
        self.energy_analysis_result: EnergyAnalysisResult | None = None
        self.opportunities = []
        self.user_interpretations = []
        self.cop_status = "Not evaluated"
        # Presentation-only global scope.  Services continue to receive an
        # explicit project/equipment/period, so this cannot alter research
        # artifacts or deterministic calculations implicitly.
        self.selected_equipment_id: str | None = None
        self.selected_period: str = "all"
        self.selected_finding_id: str | None = None
        self.agent_controller = AgentController(self)

    def reload_llm(self) -> None:
        """Apply saved model settings without requiring an application restart."""
        self.llm_manager = LLMManager(self.settings)
        self.opportunities_service = OpportunityService(self.llm_manager)

    def reset_project(self):
        self.current_project = None
        self.dataframe = None
        self.semantic_result = None
        self.import_metadata = {}
        self.data_notice = None
        self.equipment = []
        self.equipment_organization = None
        self.diagnosis_result = None
        self.energy_analysis_result = None
        self.opportunities = []
        self.user_interpretations = []
        self.selected_equipment_id = None
        self.selected_period = "all"
        self.selected_finding_id = None

    def open_project(self, project_id: str):
        project = self.projects.get(project_id)
        if not project:
            raise KeyError(project_id)
        self.current_project = project
        self.selected_equipment_id = None
        self.selected_finding_id = None
        try:
            self.dataframe = self.timeseries.load(project_id)
        except FileNotFoundError:
            self.dataframe = None
            self.data_notice = "legacy_project_data_missing" if project.source_files else None
        else:
            self.data_notice = None
        self.import_metadata = self.projects.get_import_metadata(project_id)
        saved = self.projects.load_semantics(project_id)
        if saved:
            counts: dict[str, int] = {}
            for item in saved:
                counts[item.raw_name] = counts.get(item.raw_name, 0) + 1
            roles = {
                x.raw_name: x.effective_label for x in saved if counts[x.raw_name] == 1
            }
            self.semantic_result = AnalysisResult(saved, roles, roles.copy())
            self.equipment_organization = self.equipment_service.organize(project_id, saved)
            self.equipment = self.equipment_organization.equipment
        else:
            self.semantic_result = None
            self.equipment = []
            self.equipment_organization = None
        self.diagnosis_result = None
        self.opportunities = []
        self.user_interpretations = []
        self.energy_analysis_result = self.energy_analysis.analyze(self.dataframe, self.semantic_result, project_id, self.import_metadata, self.equipment_organization) if self.dataframe is not None and self.semantic_result else None
        # Product-owned BEMS cache remains untouched.  Only stale derived
        # semantics are rebuilt after an equipment-resolution pipeline update.
        if self.dataframe is not None and saved and any(item.algorithm_version != PIPELINE_VERSION for item in saved):
            self.run_semantics()
        # Derived results are rebuilt only by the formal project analysis stack
        # from persisted raw data and persisted semantic mappings.  This makes a
        # reopened project immediately usable by every page and the Agent.
        if self.dataframe is not None and self.semantic_result is not None and self.current_project.analysis_summary.get("status") == "current":
            self.ensure_analysis_results()

    def ensure_project_loaded(self, project_id: str) -> None:
        """Synchronize a caller with the formal, selected project context."""
        if self.current_project is None or self.current_project.project_id != project_id:
            self.open_project(project_id)
        elif self.dataframe is None:
            self.open_project(project_id)

    def ensure_analysis_results(self) -> DiagnosisResult | None:
        """Restore canonical derived results from persisted project inputs.

        This is deliberately owned by ApplicationContext, not the Agent.  UI,
        dashboard, diagnostics and Agent therefore consume identical objects.
        """
        if self.diagnosis_result is not None:
            return self.diagnosis_result
        if self.dataframe is None or self.semantic_result is None or self.current_project is None:
            return None
        self.diagnosis_result = self.diagnosis.diagnose_project(
            self.dataframe, self.semantic_result, self.current_project.project_id,
            self.import_metadata, organization=self.equipment_organization,
        )
        self.equipment_organization = self.diagnosis_result.equipment
        self.equipment = self.equipment_organization.equipment if self.equipment_organization else []
        self.energy_analysis_result = self.energy_analysis.analyze(
            self.dataframe, self.semantic_result, self.current_project.project_id,
            self.import_metadata, self.equipment_organization, self.diagnosis_result.analytics,
        )
        # Restoring data must never trigger an LLM request or create a second
        # interpretation of the findings.
        self.opportunities = OpportunityService(enable_llm=False).identify(self.diagnosis_result.findings)
        self.user_interpretations = [self.interpretation.interpret(item, language=self.settings.language) for item in self.diagnosis_result.findings]
        self.diagnosis_result.consistency_validation = validate_analysis(
            self.diagnosis_result.analytics, self.diagnosis_result.findings, self.opportunities, self.energy_analysis_result,
        )
        return self.diagnosis_result

    def agent_project_state(self, project_id: str, require_analysis: bool = False) -> dict:
        """Single read model for all Agent tools, backed by persisted storage."""
        self.ensure_project_loaded(project_id)
        if require_analysis:
            self.ensure_analysis_results()
        project = self.current_project
        assert project is not None
        persisted = self.timeseries.exists(project_id)
        data_available = bool(persisted and self.dataframe is not None and not self.dataframe.empty)
        if self.energy_analysis_result is None and data_available and self.semantic_result is not None:
            self.energy_analysis_result = self.energy_analysis.analyze(
                self.dataframe, self.semantic_result, project_id, self.import_metadata, self.equipment_organization,
            )
        return {
            "project": project, "dataframe": self.dataframe, "data_available": data_available,
            "metadata": self.import_metadata, "semantics": self.semantic_result,
            "equipment": self.equipment_organization, "diagnosis": self.diagnosis_result,
            "energy": self.energy_analysis_result, "opportunities": self.opportunities,
            "user_interpretations": self.user_interpretations,
        }

    def import_data(self, path: str, sheet: str | None = None, *, mode: str = "add", allow_duplicate: bool = False) -> ImportResult:
        if not self.current_project:
            raise RuntimeError("No current project")
        result = self.importer.load(path, sheet)
        if mode not in {"add", "replace"}:
            raise ValueError(f"Unsupported import mode: {mode}")
        existing_metadata = self.projects.get_import_metadata(self.current_project.project_id)
        existing_imports = list(existing_metadata.get("imports", []))
        if mode == "replace":
            existing_imports = []
            self.project_data.clear(self.current_project.project_id)
            self.timeseries.clear(self.current_project.project_id)
        record = self.project_data.copy_import(self.current_project.project_id, path, existing_imports, allow_duplicate=allow_duplicate)
        if mode == "add" and self.timeseries.exists(self.current_project.project_id):
            previous = self.timeseries.load(self.current_project.project_id)
            combined = pd.concat([previous, result.dataframe], ignore_index=True, sort=False)
        else:
            combined = result.dataframe
        self.dataframe = combined
        self.projects.clear_runtime_results(self.current_project.project_id)
        self.timeseries.save(self.current_project.project_id, combined)
        self.import_metadata = result.metadata.to_dict()
        self.import_metadata.update({
            "source_file": record["original_filename"], "imports": [*existing_imports, record],
            "data_revision": self.current_project.data_revision + 1, "dataset_rows": len(combined),
            "dataset_columns": len(combined.columns), "mode": mode,
        })
        self.projects.save_import_metadata(self.current_project.project_id, self.import_metadata)
        if record["original_filename"] not in self.current_project.source_files:
            self.current_project.source_files.append(record["original_filename"])
        self.current_project.data_revision += 1; self.current_project.data_status = "ready"
        self.current_project.semantic_summary = {"status": "stale", "data_revision": self.current_project.data_revision}
        self.current_project.analysis_summary = {"status": "stale", "data_revision": self.current_project.data_revision}
        self.current_project.time_range = {"start": self.import_metadata.get("start"), "end": self.import_metadata.get("end")}
        self.semantic_result = None; self.equipment = []; self.equipment_organization = None; self.diagnosis_result = None; self.energy_analysis_result = None; self.opportunities = []; self.user_interpretations = []
        self.projects.save(self.current_project)
        return result

    def clear_project_data(self) -> None:
        if not self.current_project:
            raise RuntimeError("No current project")
        project_id = self.current_project.project_id
        self.project_data.clear(project_id)
        self.timeseries.clear(project_id)
        self.projects.clear_runtime_results(project_id, clear_import_metadata=True)
        self.current_project.data_revision += 1; self.current_project.data_status = "empty"
        self.current_project.source_files = []; self.current_project.time_range = {}
        self.current_project.semantic_summary = {"status": "stale", "data_revision": self.current_project.data_revision}
        self.current_project.analysis_summary = {"status": "stale", "data_revision": self.current_project.data_revision}
        self.dataframe = None; self.semantic_result = None; self.equipment = []; self.equipment_organization = None
        self.diagnosis_result = None; self.energy_analysis_result = None; self.opportunities = []; self.user_interpretations = []; self.import_metadata = {}
        self.projects.save(self.current_project)

    def run_semantics(self):
        if self.dataframe is None or not self.current_project:
            raise RuntimeError("Project data is not loaded")
        self.semantic_result = self.semantics.analyze_dataframe(
            self.dataframe,
            source_file=self.import_metadata.get("source_file"),
            sheet=self.import_metadata.get("sheet"),
            project_id=self.current_project.project_id,
        )
        self.projects.save_semantics(
            self.current_project.project_id, self.semantic_result.semantic_results
        )
        self.equipment_organization = self.equipment_service.organize(
            self.current_project.project_id, self.semantic_result.semantic_results
        )
        self.equipment = self.equipment_organization.equipment
        self.energy_analysis_result = self.energy_analysis.analyze(self.dataframe, self.semantic_result, self.current_project.project_id, self.import_metadata, self.equipment_organization)
        counts = {"ACCEPT": 0, "REVIEW": 0, "ABSTAIN": 0}
        for item in self.semantic_result.semantic_results:
            counts[item.status.value] += 1
        counts.update({"status": "current", "data_revision": self.current_project.data_revision})
        self.current_project.semantic_summary = counts
        self.projects.save(self.current_project)

    def run_diagnosis(self, progress_callback=None):
        if self.dataframe is None or self.semantic_result is None or not self.current_project:
            raise RuntimeError("Import data and run semantic analysis before diagnosis")
        emit = progress_callback or (lambda *args: None)
        emit("load_project_data", "running", None, 0, 0)
        emit("load_project_data", "completed", None, 1, 1)
        emit("semantic_mapping", "running", None, 0, 0)
        emit("semantic_mapping", "completed", None, len(self.semantic_result.semantic_results), len(self.semantic_result.semantic_results))
        self.diagnosis_result = self.diagnosis.diagnose_project(
            self.dataframe, self.semantic_result, self.current_project.project_id, self.import_metadata, emit,
        )
        self.equipment_organization = self.diagnosis_result.equipment
        self.equipment = self.equipment_organization.equipment if self.equipment_organization else []
        self.energy_analysis_result = self.energy_analysis.analyze(self.dataframe, self.semantic_result, self.current_project.project_id, self.import_metadata, self.equipment_organization, self.diagnosis_result.analytics)
        emit("energy_opportunities", "running", None, 0, len(self.diagnosis_result.findings))
        self.opportunities = self.opportunities_service.identify(self.diagnosis_result.findings, emit)
        self.user_interpretations = [self.interpretation.interpret(item, language=self.settings.language) for item in self.diagnosis_result.findings]
        emit("energy_opportunities", "completed", None, len(self.opportunities), len(self.diagnosis_result.findings))
        emit("validation", "running", None, 0, 0)
        validation = validate_analysis(self.diagnosis_result.analytics, self.diagnosis_result.findings, self.opportunities, self.energy_analysis_result)
        self.diagnosis_result.consistency_validation = validation
        if not validation.passed:
            emit("validation", "failed", None, len(validation.issues), len(validation.issues))
            self.current_project.analysis_summary = {"status": "validation_failed", "validation": validation.to_dict()}
            self.projects.save(self.current_project)
            raise RuntimeError("Analysis consistency validation failed")
        emit("validation", "completed", None, 1, 1)
        emit("finalize", "running", None, 0, 0)
        self.current_project.analysis_summary = {
            "findings": len(self.diagnosis_result.findings),
            "opportunities": len(self.opportunities),
            "skipped": list(self.diagnosis_result.skipped),
            "status": "current", "data_revision": self.current_project.data_revision,
            "validation": validation.to_dict(),
        }
        self.projects.save(self.current_project)
        emit("finalize", "completed", None, 1, 1)
        return self.diagnosis_result

    def save_review(self, point_id: str, human_label: str, note: str, equipment_id: str | None = None):
        if not self.current_project:
            raise RuntimeError("No current project")
        equipment_id = normalize_equipment_id(equipment_id) or equipment_id
        self.projects.save_review(self.current_project.project_id, point_id, human_label, note, equipment_id)
        item = self.projects.get_semantic(self.current_project.project_id, point_id)
        if item:
            item.human_verified = True; item.human_label = human_label; item.confirmed_equipment_id = equipment_id or item.equipment_id
            item.confirmed_label = human_label; item.confirmation_source = "human_confirmed"
            self.confirmed_mappings.save(self.current_project.project_id, item, note)
        self.open_project(self.current_project.project_id)
