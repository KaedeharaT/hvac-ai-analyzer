"""Presentation read models for the product UX.

The helpers in this module never calculate engineering results.  They only
organise outputs already produced by the deterministic services so the UI,
reports and tests share the same wording and evidence boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Iterable

from building_ai.models import DiagnosticFinding


@dataclass(frozen=True, slots=True)
class AnalysisReadiness:
    capability: str
    status: str
    reason: str = ""
    ready_equipment: int | None = None
    total_equipment: int | None = None


@dataclass(frozen=True, slots=True)
class PassedCheck:
    equipment_id: str
    equipment_name: str
    rule_id: str
    title: str


RULES = (
    ("low_heat_source_cop", "COP", "COP diagnosis skipped"),
    ("low_chilled_water_delta_t", "Chilled-water ΔT", "Delta-T diagnosis skipped"),
    ("off_hour_operation", "Off-hour operation", "Off-hour diagnosis skipped"),
    ("low_load_high_power", "Low-load / high-power operation", "Low-load diagnosis skipped"),
)


def analysis_readiness(energy_result, diagnosis_result=None) -> list[AnalysisReadiness]:
    """Return explicit capability states without manufacturing availability."""
    if energy_result is None:
        return [AnalysisReadiness("project_data", "unavailable", "No analyzed project data")]
    details = getattr(energy_result, "capability_details", {}) or {}
    rows: list[AnalysisReadiness] = []
    for capability in (
        "energy_consumption", "power_trend", "temperature_trend", "delta_t",
        "cop", "daily_profile", "heatmap", "weather_correlation",
        "equipment_ranking", "period_comparison",
    ):
        item = details.get(capability, {})
        available = bool(item.get("available"))
        rows.append(AnalysisReadiness(
            capability, "ready" if available else "unavailable",
            "" if available else str(item.get("reason") or "Required signals are unavailable"),
        ))
    if diagnosis_result and getattr(diagnosis_result, "analytics", None):
        kpis = diagnosis_result.analytics.equipment_kpis
        ready = sum(item.status == "available" for item in kpis)
        rows.append(AnalysisReadiness("equipment_kpi", "ready" if ready else "unavailable", "", ready, len(kpis)))
    return rows


def passed_checks(diagnosis_result) -> list[PassedCheck]:
    """Identify rules that were evaluated and did not trigger.

    The rule runner records an explicit reason whenever a rule is skipped.
    Therefore a rule is shown as passed only when the equipment KPI was
    available, there is no matching finding, and no matching skip record.
    """
    if not diagnosis_result or not getattr(diagnosis_result, "analytics", None):
        return []
    findings = {
        (item.equipment_id, item.finding_type)
        for item in diagnosis_result.findings
    }
    skipped = tuple(str(item) for item in diagnosis_result.skipped)
    rows: list[PassedCheck] = []
    for kpi in diagnosis_result.analytics.equipment_kpis:
        if kpi.status != "available":
            continue
        prefix = f"{kpi.equipment_name}: "
        equipment_skips = [item[len(prefix):] for item in skipped if item.startswith(prefix)]
        for rule_id, title, skip_marker in RULES:
            if (kpi.equipment_id, rule_id) in findings:
                continue
            if any(skip_marker in item for item in equipment_skips):
                continue
            rows.append(PassedCheck(kpi.equipment_id, kpi.equipment_name, rule_id, title))
    return rows


def finding_evidence_summary(finding: DiagnosticFinding) -> str:
    if not finding.evidence:
        return "No supporting metric was recorded."
    evidence = finding.evidence[0]
    values = evidence.metric_value
    if isinstance(values, dict):
        parts = []
        for key, value in values.items():
            if isinstance(value, float):
                display = f"{value:.1%}" if "ratio" in key else f"{value:,.3g}"
            else:
                display = str(value)
            parts.append(f"{key.replace('_', ' ')}: {display}")
        value_text = "; ".join(parts[:6])
    else:
        value_text = str(values)
    scope = finding.affected_period or {}
    period = f"{scope.get('start') or '—'} to {scope.get('end') or '—'}"
    return f"{value_text} · n={finding.valid_sample_count} · {period}"


class AnalysisReportService:
    """Export the current deterministic analysis as Markdown or HTML."""

    @staticmethod
    def markdown(project, energy_result, diagnosis_result, opportunities: Iterable = ()) -> str:
        summary = getattr(energy_result, "summary", {}) or {}
        findings = list(getattr(diagnosis_result, "findings", []) or [])
        opportunities = list(opportunities or [])
        period = (
            f"{getattr(energy_result, 'start', None) or '—'} to "
            f"{getattr(energy_result, 'end', None) or '—'}"
        )
        lines = [
            "# BuildingAI Analysis Report", "",
            f"**Project:** {getattr(project, 'name', '—')}",
            f"**Period:** {period}",
            "", "## Data and analysis readiness", "",
        ]
        for item in analysis_readiness(energy_result, diagnosis_result):
            suffix = f" — {item.reason}" if item.reason else ""
            coverage = f" ({item.ready_equipment}/{item.total_equipment} equipment)" if item.ready_equipment is not None else ""
            lines.append(f"- {item.capability.replace('_', ' ').title()}: {item.status.title()}{coverage}{suffix}")
        lines.extend([
            "", "## Energy summary", "",
            f"- Total energy: {_value(summary.get('total_energy_kwh'), 'kWh', 1)}",
            f"- Peak power: {_value(summary.get('peak_power_kw'), 'kW', 1)}",
            f"- Average COP: {_value(summary.get('average_cop'), '', 2)}",
            f"- Average ΔT: {_value(summary.get('average_delta_t_c'), '°C', 2)}",
            "", "## Equipment performance", "",
        ])
        kpis = list(getattr(getattr(diagnosis_result, "analytics", None), "equipment_kpis", []) or [])
        if not kpis:
            lines.append("No equipment KPI is available for the current evidence.")
        for item in kpis:
            metrics = item.metric_summary or {}
            lines.extend([
                f"### {item.equipment_name}", "",
                f"- Analysis status: {item.status}",
                f"- Valid samples: {item.valid_sample_count}",
                f"- Average power: {_metric(metrics, 'power_kw', 'kW')}",
                f"- Average COP: {_metric(metrics, 'cop', '')}",
                f"- Average ΔT: {_metric(metrics, 'delta_t_c', '°C')}", "",
            ])
        lines.extend([
            "", "## Deterministic findings", "",
        ])
        if not findings:
            lines.append("No deterministic finding was triggered for the analyzed evidence. This is not a statement that every condition is healthy.")
        for item in findings:
            lines.extend([
                f"### {item.title}", "",
                f"- Equipment ID: {item.equipment_id or '—'}",
                f"- Severity: {item.severity}",
                f"- What happened: {item.description}",
                f"- Project evidence: {finding_evidence_summary(item)}", "",
            ])
        lines.extend(["## Recommendations", ""])
        if not opportunities:
            lines.append("No recommendation was generated because no evidence-backed finding was available.")
        for item in opportunities:
            lines.extend([
                f"### {item.title}", "",
                f"- Recommended check: {item.recommendation}",
                f"- Expected direction: {item.expected_impact}",
                "- Verification: Re-run the same measured KPI and diagnostic rule after site review or intervention.",
                "- Energy impact: Not available unless a supported measured impact calculation exists.", "",
            ])
        lines.extend([
            "## Reference material", "",
            "This deterministic export does not attach external references automatically. "
            "Reference material used by an AI investigation must remain separately identified by source and URL.", "",
            "## Evidence boundary", "",
            "Project findings and KPI values above come from deterministic BuildingAI services. "
            "Knowledge references may explain possible causes or checks, but do not create project findings.",
        ])
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def html(project, energy_result, diagnosis_result, opportunities: Iterable = ()) -> str:
        markdown = AnalysisReportService.markdown(project, energy_result, diagnosis_result, opportunities)
        blocks: list[str] = []
        in_list = False
        for line in markdown.splitlines():
            if line.startswith("- "):
                if not in_list:
                    blocks.append("<ul>"); in_list = True
                blocks.append(f"<li>{escape(line[2:])}</li>")
                continue
            if in_list:
                blocks.append("</ul>"); in_list = False
            if line.startswith("### "):
                blocks.append(f"<h3>{escape(line[4:])}</h3>")
            elif line.startswith("## "):
                blocks.append(f"<h2>{escape(line[3:])}</h2>")
            elif line.startswith("# "):
                blocks.append(f"<h1>{escape(line[2:])}</h1>")
            elif line.startswith("**") and ":**" in line:
                label, value = line[2:].split(":**", 1)
                blocks.append(f"<p><strong>{escape(label)}:</strong>{escape(value)}</p>")
            elif line:
                blocks.append(f"<p>{escape(line)}</p>")
        if in_list:
            blocks.append("</ul>")
        return "<!doctype html><html><head><meta charset='utf-8'><title>BuildingAI Analysis Report</title></head><body>" + "".join(blocks) + "</body></html>"


def _value(value, unit: str, digits: int) -> str:
    if value is None:
        return "Not available"
    return f"{float(value):,.{digits}f} {unit}".strip()


def _metric(metrics: dict, key: str, unit: str) -> str:
    value = (metrics.get(key) or {}).get("mean")
    return _value(value, unit, 2)
