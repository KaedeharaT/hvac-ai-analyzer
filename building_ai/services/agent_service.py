from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from building_ai.llm.agent import ToolRegistry
from building_ai.storage import ProjectStore, TimeseriesStore


class AgentService:
    """Structured tool layer. LLM explanation can be added without exposing raw frames."""

    def __init__(self, projects: ProjectStore, timeseries: TimeseriesStore, analysis_getter: Callable[[], Any] | None = None, equipment_getter: Callable[[], Any] | None = None, energy_getter: Callable[[], Any] | None = None, project_state_getter: Callable[[str, bool], dict] | None = None, drawings=None):
        self.projects = projects
        self.timeseries = timeseries
        self.analysis_getter = analysis_getter
        self.equipment_getter = equipment_getter
        self.energy_getter = energy_getter
        self.project_state_getter = project_state_getter
        self.drawings = drawings
        self.tools = ToolRegistry()
        self.tools.register("list_projects", self.list_projects)
        self.tools.register("get_project_summary", self.get_project_summary)
        self.tools.register("get_semantic_mapping", self.get_semantic_mapping)
        self.tools.register("get_point_timeseries", self.get_point_timeseries)
        self.tools.register("get_analysis_results", self.get_analysis_results)
        self.tools.register("get_equipment_kpis", self.get_equipment_kpis)
        self.tools.register("get_equipment_summary", self.get_equipment_summary)
        self.tools.register("get_diagnostic_findings", self.get_diagnostic_findings)
        self.tools.register("get_energy_opportunities", self.get_energy_opportunities)
        self.tools.register("get_energy_summary", self.get_energy_summary)
        self.tools.register("get_energy_timeseries", self.get_energy_timeseries)
        self.tools.register("get_temperature_summary", self.get_temperature_summary)
        self.tools.register("list_project_drawings", self.list_project_drawings)
        self.tools.register("get_drawing_detections", self.get_drawing_detections)
        self.tools.register("get_drawing_summary", self.get_drawing_summary)
        self.tools.register("get_equipment_drawing_location", self.get_equipment_drawing_location)

    def _state(self, project_id: str, require_analysis: bool = False) -> dict | None:
        return self.project_state_getter(project_id, require_analysis) if self.project_state_getter else None

    def list_projects(self):
        return [{"project_id": p.project_id, "name": p.name} for p in self.projects.list()]

    def get_project_summary(self, project_id: str):
        state = self._state(project_id)
        project = state["project"] if state else self.projects.get(project_id)
        if not project:
            raise KeyError(project_id)
        if state is None:
            return project.to_dict()
        metadata, frame, equipment, energy = state["metadata"], state["dataframe"], state["equipment"], state["energy"]
        return {
            "project_id": project.project_id, "project_name": project.name,
            "data_available": state["data_available"], "imported_files": [item.get("original_filename") for item in metadata.get("imports", [])] or project.source_files,
            "time_range": {"start": metadata.get("start") or project.time_range.get("start"), "end": metadata.get("end") or project.time_range.get("end")},
            "sampling_interval": metadata.get("sampling_interval") or (f"{energy.sampling_interval_minutes:g} min" if energy and energy.sampling_interval_minutes else None),
            "number_of_points": int(frame.shape[1]) if frame is not None else 0,
            "number_of_rows": int(frame.shape[0]) if frame is not None else 0,
            "discovered_equipment": [binding.equipment.name for binding in equipment.heat_sources] if equipment else [],
            "available_analyses": [key for key, value in energy.capabilities.items() if value] if energy else [],
            "analysis_capabilities": energy.capability_details if energy else {},
            "latest_analysis_status": project.analysis_summary.get("status", "not_run"),
        }

    def get_semantic_mapping(self, project_id: str):
        self._state(project_id)
        return [{
            "point_id": x.point_id, "raw_name": x.raw_name,
            "source_file": x.source_file, "sheet": x.sheet,
            "ai_label": x.canonical_label,
            "effective_label": x.effective_label, "status": x.status.value,
        } for x in self.projects.load_semantics(project_id)]

    def get_point_timeseries(self, project_id: str, point_id: str, limit: int = 100):
        state = self._state(project_id)
        mapping = self.projects.get_semantic(project_id, point_id)
        if not mapping:
            raise KeyError(point_id)
        if mapping.status.value == "ABSTAIN" and not mapping.human_verified:
            raise ValueError("Point semantics are ABSTAIN; human verification is required")
        df = self.timeseries.load(project_id)
        if mapping.raw_name not in df.columns:
            raise KeyError(mapping.raw_name)
        result = {
            "point_id": point_id, "raw_name": mapping.raw_name,
            "source_file": mapping.source_file, "sheet": mapping.sheet,
            "values": df[mapping.raw_name].tail(limit).tolist(),
        }
        time_column = state["metadata"].get("time_column") if state else None
        if time_column in df.columns:
            result["timestamps"] = [str(value) for value in df[time_column].tail(limit).tolist()]
        return result

    def get_analysis_results(self, project_id: str):
        state = self._state(project_id, True)
        diagnosis = state["diagnosis"] if state else (self.analysis_getter() if self.analysis_getter else None)
        if diagnosis is None:
            return {"available": False, "reason": "Analysis has not been run for the current project."}
        return {
            "available": True,
            "consistency_validation": diagnosis.consistency_validation.to_dict() if diagnosis.consistency_validation else None,
            "equipment_kpis": [{
                "equipment": item.equipment_name, "equipment_id": item.equipment_id,
                "status": item.status, "reason": item.reason,
                "valid_sample_count": item.valid_sample_count,
                "metrics": item.metric_summary,
                "calculation_metadata": item.calculation_metadata,
            } for item in diagnosis.analytics.equipment_kpis],
        }

    def get_equipment_kpis(self, project_id: str, equipment_name: str | None = None):
        result = self.get_analysis_results(project_id)
        if not result.get("available") or not equipment_name:
            return result
        return {**result, "equipment_kpis": [x for x in result["equipment_kpis"] if x["equipment"].casefold() == equipment_name.casefold()]}

    def get_equipment_summary(self, project_id: str, equipment_name: str | None = None):
        state = self._state(project_id)
        organization = state["equipment"] if state else (self.equipment_getter() if self.equipment_getter else None)
        if organization is None:
            return []
        return [{"equipment": binding.equipment.name, "status": binding.status, "reason": binding.reason, "points": {role: point.raw_name for role, point in binding.points_by_role.items()}} for binding in organization.heat_sources if not equipment_name or binding.equipment.name.casefold() == equipment_name.casefold()]

    def get_diagnostic_findings(self, project_id: str, equipment_name: str | None = None):
        state = self._state(project_id, True)
        diagnosis = state["diagnosis"] if state else (self.analysis_getter() if self.analysis_getter else None)
        if diagnosis is None:
            return []
        names = {item.equipment_id: item.equipment_name for item in diagnosis.analytics.equipment_kpis}
        return [finding.to_dict() | {"equipment_name": names.get(finding.equipment_id)} for finding in diagnosis.findings if not equipment_name or names.get(finding.equipment_id, "").casefold() == equipment_name.casefold()]

    def get_energy_opportunities(self, project_id: str, equipment_name: str | None = None):
        """Return only formal, finding-linked opportunities for the selected project.

        This deliberately exposes the result created by the product analysis
        pipeline; it never asks the Agent to synthesize a project opportunity.
        """
        state = self._state(project_id, True)
        diagnosis = state["diagnosis"] if state else None
        opportunities = state["opportunities"] if state else []
        if diagnosis is None:
            return {"available": False, "reason": "Analysis has not been run for the current project.", "opportunities": []}
        names = {item.equipment_id: item.equipment_name for item in diagnosis.analytics.equipment_kpis}
        rows = []
        for opportunity in opportunities:
            if equipment_name and names.get(opportunity.equipment_id, "").casefold() != equipment_name.casefold():
                continue
            rows.append(opportunity.to_dict() | {"equipment_name": names.get(opportunity.equipment_id)})
        return {"available": True, "opportunities": rows}

    def get_energy_summary(self, project_id: str):
        state = self._state(project_id, True)
        result = state["energy"] if state else (self.energy_getter() if self.energy_getter else None)
        return result.to_dict() if result else {"available": False, "reason": "Energy analysis has not been run for the current project."}

    def get_energy_timeseries(self, project_id: str, chart: str = "energy_trend", range_key: str = "all"):
        state = self._state(project_id, True)
        result = state["energy"] if state else (self.energy_getter() if self.energy_getter else None)
        if result is None: return {"available": False, "reason": "Energy analysis has not been run for the current project."}
        if chart not in result.charts: return {"available": False, "reason": "Current data cannot reliably provide this analysis."}
        payload = result.charts[chart]
        if range_key in {"24h", "7d", "30d"} and payload.get("series"):
            hours = {"24h": 24, "7d": 168, "30d": 720}[range_key]
            end = pd.Timestamp(result.end)
            start = end - pd.Timedelta(hours=hours)
            scoped = []
            for series in payload["series"]:
                data = [point for point in series.get("data", []) if pd.Timestamp(point["time"]) >= start]
                scoped.append({**series, "data": data})
            payload = {**payload, "series": scoped, "requested_range": range_key, "start": start.isoformat(), "end": end.isoformat()}
        return {"available": True, "chart": chart, "result": payload}

    def get_temperature_summary(self, project_id: str):
        state = self._state(project_id, True)
        result = state["energy"] if state else (self.energy_getter() if self.energy_getter else None)
        if result is None: return {"available": False, "reason": "Energy analysis has not been run for the current project."}
        return {"available": bool(result.temperature_series), "points": [{"name": x.name, "unit": x.unit, "equipment": x.equipment_name} for x in result.temperature_series]}

    def list_project_drawings(self, project_id: str):
        self._state(project_id); return self.drawings.list_drawings(project_id) if self.drawings else []

    def get_drawing_detections(self, project_id: str):
        self._state(project_id); return self.drawings.list_detections(project_id) if self.drawings else []

    def get_drawing_summary(self, project_id: str):
        self._state(project_id); return self.drawings.summary(project_id) if self.drawings else {"drawings": 0, "detections": 0, "review_status_counts": {}}

    def get_equipment_drawing_location(self, project_id: str, equipment_id: str):
        self._state(project_id)
        locations = self.drawings.equipment_location(project_id, equipment_id) if self.drawings else []
        return {"equipment_id": equipment_id, "locations": locations, "reliable": bool(locations), "reason": None if locations else "No confirmed drawing association exists for this equipment."}
