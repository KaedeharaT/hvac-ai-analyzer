from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from building_ai.core.cop_engine import CopResult, calculate_heat_source_cop
from building_ai.models import AnalysisEvidence, AnalysisResult
from building_ai.services.equipment_service import EquipmentOrganization, EquipmentService, HeatSourceBinding


@dataclass(slots=True)
class EquipmentAnalysisResult:
    """Single source of truth for one equipment's validated analysis result."""
    equipment_id: str
    equipment_name: str
    status: str
    reason: str
    binding: HeatSourceBinding
    cop: CopResult | None = None
    load_ratio: pd.Series | None = None
    load_mode: str | None = None
    operating: pd.Series | None = None
    timestamps: pd.Series | None = None
    evidence: list[AnalysisEvidence] = field(default_factory=list)
    valid_mask: pd.Series | None = None
    metric_summary: dict[str, dict[str, float | int | None]] = field(default_factory=dict)
    calculation_metadata: dict[str, str] = field(default_factory=dict)
    validation: list[str] = field(default_factory=list)

    @property
    def valid_sample_count(self) -> int:
        return int(self.metric_summary.get("cop", {}).get("valid_count") or 0)


# Product compatibility name retained for callers created before the formal
# equipment result object was introduced.
EquipmentKPI = EquipmentAnalysisResult


@dataclass(slots=True)
class AnalyticsResult:
    equipment_kpis: list[EquipmentAnalysisResult] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def available_kpis(self) -> list[EquipmentAnalysisResult]:
        return [item for item in self.equipment_kpis if item.status == "available"]


class AnalyticsService:
    """Deterministic KPI calculation using EquipmentService bindings only."""

    def analyze_project(
        self, df: pd.DataFrame, semantics: AnalysisResult, project_id: str,
        import_metadata: dict[str, Any] | None = None,
        organization: EquipmentOrganization | None = None,
        progress_callback: Callable[[str, str, str | None, int, int], None] | None = None,
    ) -> AnalyticsResult:
        organization = organization or EquipmentService().organize(project_id, semantics.semantic_results)
        timestamps = self._timestamps(df, import_metadata or {})
        result = AnalyticsResult()
        total = len(organization.heat_sources)
        for index, binding in enumerate(organization.heat_sources, start=1):
            if progress_callback:
                progress_callback("calculate_kpi", "running", binding.equipment.name, index, total)
            try:
                kpi = self._analyze_heat_source(df, binding, timestamps)
            except Exception as exc:  # A bad device must not abort the project-wide analysis.
                kpi = EquipmentKPI(
                    binding.equipment_id, binding.equipment.name, "skipped",
                    "KPI calculation failed; review this equipment's mapped signals", binding,
                )
                result.skipped.append(f"{binding.equipment.name}: {exc}")
            result.equipment_kpis.append(kpi)
            if progress_callback:
                status = "completed" if kpi.status == "available" else "warning"
                progress_callback("calculate_kpi", status, binding.equipment.name, index, total)
        if not organization.heat_sources:
            result.skipped.append("No heat-source points with usable semantic mappings")
        return result

    def calculate_cop(self, df: pd.DataFrame, semantics: AnalysisResult) -> CopResult:
        """Compatibility helper for old callers; never selects ambiguous columns."""
        organization = EquipmentService().organize("<unbound-project>", semantics.semantic_results)
        ready = [item for item in organization.heat_sources if item.status == "ready"]
        if len(ready) != 1:
            reason = "No unique heat-source equipment binding" if not ready else "Multiple heat-source bindings; analyze by equipment"
            return CopResult(False, reason)
        return self._calculate_binding_cop(df, ready[0])

    def _analyze_heat_source(
        self, df: pd.DataFrame, binding: HeatSourceBinding, timestamps: pd.Series | None
    ) -> EquipmentAnalysisResult:
        if binding.status != "ready":
            return EquipmentKPI(binding.equipment_id, binding.equipment.name, "skipped", binding.reason, binding)
        cop = self._calculate_binding_cop(df, binding)
        if not cop.available:
            return EquipmentKPI(binding.equipment_id, binding.equipment.name, "skipped", cop.reason, binding, cop=cop)

        roles = binding.points_by_role
        operating = cop.input_power_kw > 0 if cop.input_power_kw is not None else pd.Series(False, index=df.index)
        capacity, capacity_reason = self._capacity_kw(df, binding)
        load_ratio: pd.Series | None = None
        load_mode: str | None = None
        if capacity is not None and cop.cooling_load_kw is not None:
            load_ratio = (cop.cooling_load_kw / capacity).where(operating & (capacity > 0))
            load_mode = "rated_capacity"
        elif cop.cooling_load_kw is not None:
            reference = cop.cooling_load_kw.where(operating).quantile(0.90)
            if pd.notna(reference) and reference > 0:
                load_ratio = (cop.cooling_load_kw / reference).where(operating)
                load_mode = "relative_load_p90"

        valid_mask = cop.series.notna() if cop.series is not None else pd.Series(False, index=df.index)
        metric_summary = self._metric_summary(cop, valid_mask)
        evidence = self._evidence(binding, cop, timestamps, valid_mask, load_ratio, load_mode, capacity_reason)
        return EquipmentKPI(
            binding.equipment_id, binding.equipment.name, "available", "Available", binding,
            cop, load_ratio, load_mode, operating, timestamps, evidence, valid_mask, metric_summary,
            {"cop_formula": "COP = thermal_load_kw / input_power_kw", "aggregation": "mean of valid timestamp COP values", "delta_t_definition": "return_temperature - supply_temperature"},
        )

    @staticmethod
    def _stats(values: pd.Series | None, valid_mask: pd.Series) -> dict[str, float | int | None]:
        if values is None:
            return {"mean": None, "median": None, "valid_count": 0}
        valid = values.where(valid_mask).dropna()
        return {
            "mean": float(valid.mean()) if not valid.empty else None,
            "median": float(valid.median()) if not valid.empty else None,
            "valid_count": int(valid.size),
        }

    @classmethod
    def _metric_summary(cls, cop: CopResult, valid_mask: pd.Series) -> dict[str, dict[str, float | int | None]]:
        return {
            "supply_temp_c": cls._stats(cop.supply_temp_c, valid_mask),
            "return_temp_c": cls._stats(cop.return_temp_c, valid_mask),
            "delta_t_c": cls._stats(cop.delta_t_c, valid_mask),
            "flow_lps": cls._stats(cop.flow_lps, valid_mask),
            "power_kw": cls._stats(cop.input_power_kw, valid_mask),
            "thermal_load_kw": cls._stats(cop.cooling_load_kw, valid_mask),
            "cop": dict(cop.summary or {}),
        }

    @staticmethod
    def _calculate_binding_cop(df: pd.DataFrame, binding: HeatSourceBinding) -> CopResult:
        roles = binding.points_by_role
        get = lambda role: roles[role]
        return calculate_heat_source_cop(
            df,
            get("heat_source_supply_temp").column or get("heat_source_supply_temp").raw_name,
            get("heat_source_return_temp").column or get("heat_source_return_temp").raw_name,
            get("heat_source_flow").column or get("heat_source_flow").raw_name,
            get("heat_source_power").column or get("heat_source_power").raw_name,
            get("heat_source_flow").unit, get("heat_source_power").unit,
            get("heat_source_supply_temp").unit,
        )

    @staticmethod
    def _timestamps(df: pd.DataFrame, metadata: dict[str, Any]) -> pd.Series | None:
        name = metadata.get("time_column")
        if name and name in df.columns:
            values = pd.to_datetime(df[name], errors="coerce")
            if values.notna().any():
                return values
        return None

    @staticmethod
    def _capacity_kw(df: pd.DataFrame, binding: HeatSourceBinding) -> tuple[float | None, str]:
        item = binding.points_by_role.get("heat_source_capacity")
        if item is None:
            return None, "rated capacity unavailable; relative load may be used"
        values = pd.to_numeric(df[item.column or item.raw_name], errors="coerce")
        unit = str(item.unit or "").lower()
        if unit == "w":
            values = values / 1000.0
        elif unit != "kw":
            return None, f"rated capacity unit unsupported or unknown: {item.unit!r}"
        capacity = values.max(skipna=True)
        return (float(capacity), "rated capacity from BEMS point") if pd.notna(capacity) and capacity > 0 else (None, "rated capacity invalid")

    @staticmethod
    def _evidence(
        binding: HeatSourceBinding, cop: CopResult, timestamps: pd.Series | None,
        valid_mask: pd.Series, load_ratio: pd.Series | None, load_mode: str | None, capacity_reason: str,
    ) -> list[AnalysisEvidence]:
        roles = binding.points_by_role
        point_ids = [item.point_id for item in roles.values() if item.point_id]
        columns = [item.column or item.raw_name for item in roles.values()]
        start = end = None
        if timestamps is not None and timestamps.notna().any():
            start, end = timestamps.min().isoformat(), timestamps.max().isoformat()
        items = [AnalysisEvidence(
            point_ids, binding.equipment_id, start, end, "heat_source_cop",
            cop.summary or {}, "ratio", {"min": 0.5, "max": 15.0}, cop.conversion_method,
            columns, [*cop.warnings, f"filtered_samples={cop.filtered_count}"],
        )]
        if cop.delta_t_c is not None:
            delta = cop.delta_t_c.where(valid_mask).dropna()
            items.append(AnalysisEvidence(
                point_ids, binding.equipment_id, start, end, "chilled_water_delta_t",
                {"mean": float(delta.mean()) if not delta.empty else None,
                 "median": float(delta.median()) if not delta.empty else None,
                 "valid_count": int(delta.size)}, "°C", None,
                "return_temperature - supply_temperature", columns,
            ))
        if load_ratio is not None:
            valid = load_ratio.dropna()
            items.append(AnalysisEvidence(
                point_ids, binding.equipment_id, start, end, "load_ratio",
                {"mean": float(valid.mean()) if not valid.empty else None,
                 "median": float(valid.median()) if not valid.empty else None,
                 "mode": load_mode}, "ratio", None, load_mode or "", columns, [capacity_reason],
            ))
        return items
