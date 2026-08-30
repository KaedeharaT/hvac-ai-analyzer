"""Deterministic opportunity mapping with optional provider-neutral explanation."""

from __future__ import annotations

from dataclasses import asdict
from uuid import NAMESPACE_URL, uuid5

from building_ai.llm import LLMManager
from building_ai.llm.prompts import ENERGY_DIAGNOSIS_SYSTEM_PROMPT, build_energy_diagnosis_prompt
from building_ai.models import DiagnosticFinding, EnergySavingOpportunity


class OpportunityService:
    def __init__(self, llm_manager: LLMManager | None = None, enable_llm: bool = True):
        self.llm_manager = llm_manager
        self.enable_llm = enable_llm

    def identify(self, findings: list[DiagnosticFinding], progress_callback=None) -> list[EnergySavingOpportunity]:
        opportunities: list[EnergySavingOpportunity] = []
        total = len(findings)
        for index, finding in enumerate(findings, start=1):
            if progress_callback:
                progress_callback("energy_opportunities", "running", finding.equipment_id, index, total)
            opportunity = self._from_finding(finding)
            if opportunity is None:
                continue
            opportunity.llm_explanation = self._explain(finding, opportunity)
            opportunities.append(opportunity)
            if progress_callback:
                progress_callback("energy_opportunities", "completed", finding.equipment_id, index, total)
        return opportunities

    @staticmethod
    def _from_finding(finding: DiagnosticFinding) -> EnergySavingOpportunity | None:
        templates = {
            "low_heat_source_cop": (
                "HVAC_OPERATION", "Improve heat-source operating efficiency",
                "Review condenser/evaporator heat exchange, chilled-water temperature setpoints, low-load operation, maintenance condition, and chiller staging. Confirm the cause on site before changing controls.",
                "Medium (requires site validation)", "medium", "P1",
            ),
            "low_chilled_water_delta_t": (
                "HYDRONIC_CONTROL", "Address low chilled-water ΔT",
                "Check for large-flow/small-ΔT operation, bypass flow, pump VFD differential-pressure control, valve behavior, and terminal heat exchange. Validate water-system conditions before adjustment.",
                "Medium (requires site validation)", "medium", "P1",
            ),
            "off_hour_operation": (
                "SCHEDULE_CONTROL", "Optimize non-work-hour operation",
                "Review equipment schedules, holiday calendars, night demand, manual overrides, and controlled shutdown sequencing. Reduce runtime only after confirming required after-hours loads.",
                "Medium", "low", "P1",
            ),
            "low_load_high_power": (
                "STAGING_CONTROL", "Improve low-load operating strategy",
                "Review equipment staging, minimum load limits, variable-speed settings, and temperature setpoints to avoid inefficient low-load operation.",
                "Medium (requires site validation)", "medium", "P2",
            ),
            "frequent_start_stop": (
                "STAGING_CONTROL", "Reduce frequent heat-source cycling",
                "Review minimum run/off timers, staging sequence, sensor stability, and control deadbands before changing the operating logic.",
                "Medium (requires site validation)", "medium", "P2",
            ),
            "parallel_heat_source_operation": (
                "STAGING_CONTROL", "Review parallel heat-source operation",
                "Review plant staging setpoints and load allocation. Confirm total load and resilience requirements before reducing the number of running units.",
                "Medium (requires site validation)", "medium", "P2",
            ),
        }
        template = templates.get(finding.finding_type)
        if not template:
            return None
        category, title, recommendation, impact, difficulty, priority = template
        return EnergySavingOpportunity(
            str(uuid5(NAMESPACE_URL, f"building-ai:opportunity:{finding.finding_id}")),
            [finding.finding_id], finding.equipment_id, category, title, recommendation,
            impact, difficulty, priority, list(finding.evidence), finding.confidence,
        )

    def _explain(
        self, finding: DiagnosticFinding, opportunity: EnergySavingOpportunity
    ) -> dict:
        if not self.enable_llm:
            return {"status": "skipped", "reason": "LLM explanation disabled"}
        manager = self.llm_manager or LLMManager()
        # Compatibility with earlier integrations that passed an LLMClient-like
        # object; all product-owned paths pass an LLMManager.
        if hasattr(manager, "get_provider"):
            provider = manager.get_provider()
            if not provider.is_configured:
                return {"status": "skipped", "reason": "LLM unavailable / not configured"}
            call = provider.generate_json
        else:
            client = manager
            if getattr(getattr(client, "settings", None), "provider", "not_configured") == "not_configured":
                return {"status": "skipped", "reason": "LLM unavailable / not configured"}
            call = lambda prompt, system_prompt, **kwargs: client.chat_json(
                prompt, system_msg=system_prompt, **kwargs
            )
        try:
            response = call(
                build_energy_diagnosis_prompt(finding.to_dict(), opportunity.to_dict()),
                system_prompt=ENERGY_DIAGNOSIS_SYSTEM_PROMPT, temperature=0, seed=0,
            )
            if not isinstance(response, dict):
                raise ValueError("LLM response was not a JSON object")
            allowed = {"summary", "confirmed_by_data", "possible_causes", "recommended_actions", "additional_data_needed"}
            return {"status": "available", **{key: response.get(key, [] if key != "summary" else "") for key in allowed}}
        except Exception as exc:
            return {"status": "unavailable", "reason": str(exc)}
