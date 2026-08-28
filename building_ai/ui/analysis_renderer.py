"""Language-aware rendering of structured analysis objects; no recomputation."""
from __future__ import annotations

from building_ai.i18n import tr


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
