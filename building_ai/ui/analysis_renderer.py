"""Language-aware rendering of structured analysis objects; no recomputation."""
from __future__ import annotations

from building_ai.i18n import tr


def localized_guidance(context):
    """Render stored recommendations without mutating research/service results."""
    from dataclasses import replace
    findings = {f.finding_id: f for f in context.diagnosis_result.findings} if context.diagnosis_result else {}
    result = []
    for item in context.user_interpretations:
        finding = findings.get(item.finding_id)
        if finding is None:
            result.append(item)
            continue
        prefix = 'guidance_' + finding.finding_type
        if tr(prefix + '_problem') == prefix + '_problem': prefix = 'guidance_other'
        actions = []
        for index in range(10):
            key = f'{prefix}_action_{index}'
            value = tr(key)
            if value == key: break
            actions.append(value)
        result.append(replace(item, problem=tr(prefix + '_problem'), explanation=tr(prefix + '_explanation'),
            actions=tuple(actions), expected_effect=tr(prefix + '_expected_effect'),
            implementation_difficulty=tr('guidance_difficulty_low_medium' if finding.finding_type == 'low_chilled_water_delta_t' else 'guidance_difficulty_medium')))
    return result


def evidence_text(finding):
    if not finding.evidence: return tr('analysis_no_evidence')
    values = finding.evidence[0].metric_value
    parts = []
    if isinstance(values, dict):
        for key, value in list(values.items())[:6]:
            label = tr('analysis_metric_' + key)
            if label == 'analysis_metric_' + key: label = key
            display = f'{value:.1%}' if isinstance(value, float) and 'ratio' in key else f'{value:,.3g}' if isinstance(value, float) else str(value)
            parts.append(f'{label}: {display}')
    else: parts.append(str(values))
    period = finding.affected_period or {}
    return '; '.join(parts) + f" · n={finding.valid_sample_count} · {period.get('start') or '—'} – {period.get('end') or '—'}"


def reason_text(reason: str) -> str:
    if reason.startswith("multiple_candidates:"):
        return tr("analysis_multiple_candidates", field=reason.split(":", 1)[1])
    if reason.startswith("missing_signals:"):
        return tr("analysis_missing_signals", fields=reason.split(":", 1)[1])
    if reason.startswith("Multiple candidates for "):
        return tr("analysis_multiple_candidates", field=reason.removeprefix("Multiple candidates for "))
    if reason.startswith("Missing "):
        return tr("analysis_missing_signals", fields=reason.removeprefix("Missing "))
    if reason in {"Available", ""}: return tr("analysis_available")
    if reason.startswith("No heat-source"): return tr("analysis_no_heat_source")
    return reason


def finding_text(finding) -> tuple[str, str]:
    key = f"finding_{finding.finding_type}"
    title = tr(f"{key}_title")
    description = tr(f"{key}_description")
    return (finding.title if title == f"{key}_title" else title,
            finding.description if description == f"{key}_description" else description)


def opportunity_text(opportunity) -> tuple[str, str]:
    key = f"opportunity_{opportunity.category.lower()}"
    title, description = tr(f"{key}_title"), tr(f"{key}_description")
    return (opportunity.title if title == f"{key}_title" else title,
            opportunity.recommendation if description == f"{key}_description" else description)


def opportunity_impact_text(opportunity) -> str:
    key = f"opportunity_{opportunity.category.lower()}_impact"
    translated = tr(key)
    return opportunity.expected_impact if translated == key else translated


def opportunity_priority_text(opportunity) -> str:
    return tr(f"opportunity_priority_{opportunity.priority.lower()}")
