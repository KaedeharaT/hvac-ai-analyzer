from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

from building_ai.config import Settings
from building_ai.llm import LLMManager
from building_ai.core.preprocessing import find_time_column, normalize_dataframe, normalize_header
from building_ai.core.research_semantics import (
    conservative_gate, deterministic_label, direct_prompt, parse_point, validate_mapping,
)
from building_ai.models import AnalysisResult, SemanticResult, make_point_id
from building_ai.storage.confirmed_mapping_store import ConfirmedMappingStore

logger = logging.getLogger(__name__)
PIPELINE_VERSION = "product-equipment-resolver-v4"


def _engineering_unit(label: str, point, values: pd.Series) -> tuple[str | None, str | None]:
    """Conservative unit recovery for common unit-less BEMS headers."""
    if point.unit:
        return point.unit, None
    if label.endswith("_temp"):
        return "°C", "engineering_unit_inferred_temperature_c"
    if label == "heat_source_power" and point.quantity == "power":
        return "kW", "engineering_unit_inferred_power_kw"
    if label == "heat_source_flow" and point.quantity == "flow":
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if not numeric.empty and numeric.quantile(0.95) >= 30:
            return "L/min", "engineering_unit_inferred_flow_l_min"
    return None, None


class SemanticService:
    """The single product entry point for semantic analysis."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.load()
        self.llm_manager = LLMManager(self.settings)
        self.confirmed_mappings = ConfirmedMappingStore(self.settings.confirmed_dataset_path)
        # Research source: paper_research V3 strict gate + V4 structured parser.

    def analyze_dataframe(
        self, dataframe: pd.DataFrame, *, source_file: str | None = None,
        sheet: str | None = None, project_id: str = "<unbound-project>",
        backend: Literal["offline", "enhanced", "research", "legacy_c1c8"] = "offline",
    ) -> AnalysisResult:
        source_columns = {normalize_header(column): str(column) for column in dataframe.columns}
        df = normalize_dataframe(dataframe)
        time_column = find_time_column(df)
        if time_column:
            df = df.drop(columns=[time_column])
        if backend == "legacy_c1c8":
            return self._run_legacy_c1c8(df, source_file, sheet, project_id)
        result = self._run_product(df, source_file, sheet, project_id, source_columns, use_llm=backend in {"enhanced", "research"})
        if time_column:
            result.warnings.append(f"timestamp_column_excluded:{time_column}")
        return result

    def _run_product(
        self, df: pd.DataFrame, source_file: str | None, sheet: str | None,
        project_id: str, source_columns: dict[str, str], *, use_llm: bool,
    ) -> AnalysisResult:
        results: list[SemanticResult] = []
        role_dict: dict[str, str] = {}
        slot_details: dict = {}
        unit_db: dict = {}
        provider = self.llm_manager.get_provider() if use_llm else None
        for col in df.columns:
            original_name = source_columns.get(str(col), str(col))
            point = parse_point(str(col))
            label, confidence, reason = deterministic_label(point)
            label, gate_reason = conservative_gate(label, str(col))
            if gate_reason:
                confidence, reason = max(confidence, 0.90), gate_reason
            prior: dict = {"source": "engineering_offline"}
            historical = self.confirmed_mappings.exact(str(col), point.unit)
            if historical:
                label, confidence, reason = historical.confirmed_label, 1.0, "confirmed_dataset_exact"
                prior = {"source": "confirmed_dataset_exact", "confirmed_at": historical.confirmed_at,
                         "source_project_id": historical.source_project_id}
            if provider and provider.is_configured:
                try:
                    response = provider.generate_json(direct_prompt(str(col)), temperature=0, seed=0)
                    proposed = str(response.get("label", "other"))
                    raw_confidence = float(response.get("confidence", 0.0))
                    if not historical and proposed in {"heat_source_supply_temp", "heat_source_return_temp", "heat_source_flow", "heat_source_power", "terminal_supply_air_temp", "terminal_return_air_temp", "terminal_power", "other"}:
                        label, gate_reason = conservative_gate(proposed, str(col))
                        confidence = max(0.0, min(1.0, raw_confidence))
                        reason = gate_reason or str(response.get("reason", "llm_direct_prior"))
                        prior = {"source": "pluggable_llm_direct", "raw_label": proposed, "response": response}
                    else:
                        prior = {"source": "pluggable_llm_direct", "error": "invalid_strict_label"}
                except Exception as exc:  # fallback is a normal no-LLM outcome
                    prior = {"source": "pluggable_llm_direct", "error": str(exc), "fallback": "engineering_offline"}
            unit, unit_reason = _engineering_unit(label, point, df[col])
            unit_confidence = 0.95 if unit else 0.0
            valid, warnings = validate_mapping(label, point, df[col])
            abstained = label == "other" and confidence == 0
            review = bool(warnings) or (label == "other" and not abstained and "insufficient" in reason)
            gate = "ABSTAIN" if abstained else ("REVIEW" if review else "ACCEPT")
            debug = {
                "backend": "research_baseline_v3_v4_product", "research_source": "paper_research/src/bems_v3/core.py; paper_research/src/bems_v4/core.py",
                "unit_confidence": unit_confidence, "gate_status": gate, "physics_warning": warnings,
                "structured_evidence": point.to_dict(), "llm_prior": prior,
            }
            item = SemanticResult(
                raw_name=original_name, column=original_name, canonical_label=label,
                point_id=make_point_id(project_id, source_file, sheet, original_name),
                canonical_name=label, equipment_type=point.equipment_type if point.equipment_type != "unknown" else None,
                # A confirmed label is reusable as a name/unit assertion.  Device
                # identity is deliberately project-local: CH-01 in two projects
                # must never become the same physical asset.
                equipment_id=point.equipment_id, confirmed_equipment_id=None,
                group_id=point.loop_context,
                relation_confidence=0.95 if point.equipment_id else 0.0,
                physical_quantity=point.quantity,
                signal_type="cumulative" if point.quantity == "energy" else "instantaneous" if point.quantity == "power" else point.measurement_type,
                medium=point.medium,
                position=point.role if point.role != "unknown" else None, unit=unit, confidence=confidence, gate_status=gate,
                physical_validity=valid, needs_review=review, abstained=abstained, suspicious=review,
                reason=reason, physics_warnings=warnings,
                debug_metadata=debug, source_file=source_file, sheet=sheet,
                confirmation_source="confirmed_dataset_exact" if historical else None,
                model_provider=provider.provider_id if provider and provider.is_configured else "offline",
                model_name=self.settings.model if provider and provider.is_configured else None,
                prompt_version="strict_8_direct_v1" if use_llm else None,
                algorithm_version=PIPELINE_VERSION,
            )
            if unit_reason:
                item.physics_warnings.append(unit_reason)
            results.append(item)
            role_dict[original_name] = label
            slot_details[str(col)] = {"per_label": {}, "debug": debug}
            unit_db[str(col)] = {"unit": unit, "confidence": unit_confidence}
        return AnalysisResult(results, role_dict, role_dict.copy(), slot_details, unit_db)

    def _run_legacy_c1c8(
        self, df: pd.DataFrame, source_file: str | None, sheet: str | None,
        project_id: str,
    ) -> AnalysisResult:
        # Compatibility path for old saved research comparisons only. New product
        # calls use _run_product and never depend on the old C1-C8 runtime.
        from building_ai.research.hvac_power_col_memory import batch_physical_role_review
        from building_ai.research.client_adapter import ResearchLLMClientAdapter
        from building_ai.research.unit_adapter import extract_research_unit_db

        role_dict, ai_roles, slot_details = batch_physical_role_review(
            df,
            client=ResearchLLMClientAdapter(self.llm_manager.get_provider()),
            role_db_path="",
        )
        results: list[SemanticResult] = []
        unit_db = extract_research_unit_db(df)
        for col in df.columns:
            detail = slot_details.get(col, {})
            debug = detail.get("debug", {})
            label = ai_roles.get(col, "other")
            unit_info = unit_db.get(str(col), {})
            unit = unit_info.get("unit")
            gate = str(debug.get("gate_status") or "")
            warnings = debug.get("physics_warning") or []
            score = debug.get("final_score")
            results.append(SemanticResult(
                raw_name=str(col), column=str(col), canonical_label=label,
                point_id=make_point_id(project_id, source_file, sheet, str(col)),
                canonical_name=label, equipment_type=debug.get("equipment_type_llm"),
                physical_quantity=debug.get("physical_type_llm"), unit=unit,
                confidence=float(score) if score is not None else None,
                gate_status=gate or None, physical_validity=debug.get("physical_valid"),
                needs_review=bool(debug.get("suspicious_flag")),
                abstained=gate.upper() == "ABSTAIN",
                suspicious=bool(debug.get("suspicious_flag")),
                reason=debug.get("explain_zh") or debug.get("semantic_reason"),
                physics_warnings=list(warnings), per_label_scores=detail.get("per_label", {}),
                debug_metadata=debug, source_file=source_file, sheet=sheet,
                model_provider=self.settings.provider, model_name=self.settings.model,
                prompt_version="direct_v1_name",
            ))
        return AnalysisResult(results, role_dict, ai_roles, slot_details, unit_db)
