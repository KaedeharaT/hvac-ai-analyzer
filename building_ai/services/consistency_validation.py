"""Cross-layer validation for the product-owned HVAC analysis result."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from building_ai.models import DiagnosticFinding, EnergySavingOpportunity
from building_ai.core.equipment_identity import normalize_equipment_id
from building_ai.services.analytics_service import AnalyticsResult


@dataclass(slots=True)
class ValidationIssue:
    code: str
    message: str
    equipment_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class AnalysisConsistencyValidation:
    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "issues": [issue.to_dict() for issue in self.issues]}


def validate_analysis(
    analytics: AnalyticsResult,
    findings: Iterable[DiagnosticFinding],
    opportunities: Iterable[EnergySavingOpportunity],
    energy: Any | None = None,
) -> AnalysisConsistencyValidation:
    """Reject cross-layer contradictions before results are presented as success."""
    issues: list[ValidationIssue] = []
    kpis = list(analytics.equipment_kpis)
    ids = [item.equipment_id for item in kpis]
    if len(ids) != len(set(ids)):
        issues.append(ValidationIssue("duplicate_equipment", "Duplicate equipment analysis result"))
    names = {item.equipment_id: item for item in kpis}
    required = {"heat_source_supply_temp", "heat_source_return_temp", "heat_source_flow", "heat_source_power"}
    for item in kpis:
        if item.status != "available":
            continue
        roles = item.binding.points_by_role
        if not required <= set(roles):
            issues.append(ValidationIssue("missing_signal", "Available KPI has incomplete signal binding", item.equipment_id)); continue
        columns = [roles[role].column or roles[role].raw_name for role in required]
        if len(columns) != len(set(columns)):
            issues.append(ValidationIssue("non_unique_signal", "A required signal is reused by more than one KPI role", item.equipment_id))
        if str(roles["heat_source_power"].unit or "").lower() not in {"kw", "w"}:
            issues.append(ValidationIssue("invalid_power_unit", "Input power unit is not W or kW", item.equipment_id))
        power = roles["heat_source_power"]
        if power.physical_quantity not in {None, "power"} or power.signal_type in {"cumulative", "energy"}:
            issues.append(ValidationIssue("power_energy_confusion", "Input power binding is not an instantaneous power signal", item.equipment_id))
        for role, point in roles.items():
            owner = normalize_equipment_id(point.effective_equipment_id)
            expected_owner = normalize_equipment_id(item.equipment_name)
            if owner and expected_owner and owner != expected_owner:
                issues.append(ValidationIssue("equipment_relation_mismatch", f"{role} belongs to a different equipment", item.equipment_id))
        if str(roles["heat_source_flow"].unit or "").lower().replace("³", "3") not in {"l/s", "l/min", "m3/h"}:
            issues.append(ValidationIssue("invalid_flow_unit", "Flow unit is not L/s, L/min, or m³/h", item.equipment_id))
        if item.cop is None or item.cop.series is None or item.valid_mask is None:
            issues.append(ValidationIssue("missing_cop", "Available KPI has no COP calculation data", item.equipment_id)); continue
        valid_count = int(item.cop.series.notna().sum())
        if item.valid_sample_count != valid_count:
            issues.append(ValidationIssue("valid_count_mismatch", "KPI valid sample count is inconsistent", item.equipment_id))
        for key, values in (("delta_t_c", item.cop.delta_t_c), ("power_kw", item.cop.input_power_kw), ("thermal_load_kw", item.cop.cooling_load_kw)):
            summary = item.metric_summary.get(key, {})
            if values is None or summary.get("mean") is None:
                issues.append(ValidationIssue("missing_metric", f"Missing {key} summary", item.equipment_id)); continue
            observed = float(values.where(item.valid_mask).dropna().mean())
            if abs(float(summary["mean"]) - observed) > 1e-8:
                issues.append(ValidationIssue("metric_summary_mismatch", f"{key} summary differs from canonical valid samples", item.equipment_id))
        cop_summary = item.metric_summary.get("cop", {})
        if not item.cop.summary or abs(float(cop_summary.get("mean", -999)) - float(item.cop.summary["mean"])) > 1e-8:
            issues.append(ValidationIssue("cop_summary_mismatch", "COP summary is inconsistent", item.equipment_id))

    finding_list = list(findings)
    seen_findings: set[tuple[str | None, str]] = set()
    for finding in finding_list:
        key = (finding.equipment_id, finding.finding_type)
        if key in seen_findings:
            issues.append(ValidationIssue("duplicate_finding", "Duplicate finding for equipment and code", finding.equipment_id))
        seen_findings.add(key)
        if finding.equipment_id not in names:
            issues.append(ValidationIssue("orphan_finding", "Finding is not linked to an analyzed equipment", finding.equipment_id))
        if not finding.evidence or finding.valid_sample_count <= 0:
            issues.append(ValidationIssue("incomplete_finding_evidence", "Finding has insufficient traceable evidence", finding.equipment_id))
        if finding.finding_type == "low_chilled_water_delta_t" and finding.equipment_id in names:
            metric = finding.evidence[0].metric_value
            expected = names[finding.equipment_id].metric_summary.get("delta_t_c", {}).get("mean")
            if isinstance(metric, dict) and expected is not None and abs(float(metric.get("mean", -999)) - float(expected)) > 1e-8:
                issues.append(ValidationIssue("delta_t_evidence_mismatch", "Finding ΔT evidence differs from KPI ΔT", finding.equipment_id))

    finding_by_id = {finding.finding_id: finding for finding in finding_list}
    for opportunity in opportunities:
        linked = [finding_by_id.get(identifier) for identifier in opportunity.related_finding_ids]
        if not linked or any(item is None for item in linked):
            issues.append(ValidationIssue("orphan_opportunity", "Opportunity has no valid finding linkage", opportunity.equipment_id)); continue
        if any(item.equipment_id != opportunity.equipment_id for item in linked):
            issues.append(ValidationIssue("opportunity_equipment_mismatch", "Opportunity equipment differs from its finding", opportunity.equipment_id))
    if energy is not None:
        project_ids = {item.binding.equipment.project_id for item in kpis}
        if project_ids and (len(project_ids) != 1 or energy.project_id not in project_ids):
            issues.append(ValidationIssue("energy_project_mismatch", "Energy result and equipment KPIs belong to different projects"))
        chart_equipment = {item.equipment_name for item in kpis}
        for code in ("delta_t_trend", "cop_trend"):
            for series in energy.charts.get(code, {}).get("series", []):
                if series.get("name") not in chart_equipment:
                    issues.append(ValidationIssue("chart_equipment_mismatch", f"{code} contains an unknown equipment", series.get("name")))
                if not series.get("data"):
                    issues.append(ValidationIssue("empty_chart_series", f"{code} is marked available with no values", series.get("name")))
        for item in energy.charts.get("equipment_ranking", {}).get("data", []):
            if item.get("name") not in chart_equipment:
                issues.append(ValidationIssue("ranking_equipment_mismatch", "Energy ranking contains an unknown equipment", item.get("name")))
        mapping = {"energy_timeseries": "energy_consumption", "power_timeseries": "power_trend", "temperature_timeseries": "temperature_trend", "daily_load_profile": "daily_profile", "load_heatmap": "heatmap", "equipment_breakdown": "equipment_ranking"}
        for legacy, canonical in mapping.items():
            if energy.capabilities.get(legacy) != energy.capabilities.get(canonical):
                issues.append(ValidationIssue("capability_alias_mismatch", f"{legacy} differs from {canonical}"))
    return AnalysisConsistencyValidation(not issues, issues)
