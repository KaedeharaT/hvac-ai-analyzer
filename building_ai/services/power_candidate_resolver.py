"""Auditable within-equipment candidate resolution for KPI point roles."""
from __future__ import annotations

from building_ai.models import SemanticResult


POWER_UNITS = {"w", "kw", "mw"}
ENERGY_UNITS = {"wh", "kwh", "mwh"}


def resolve_role(role: str, candidates: list[SemanticResult]) -> tuple[SemanticResult | None, dict]:
    """Select one engineering-supported candidate or retain true ambiguity.

    Resolution never crosses equipment groups.  For power, cumulative/auxiliary
    signals are explicitly penalised.  Other roles only use confirmation and
    semantic confidence as tie-breakers.
    """
    ranked: list[tuple[float, SemanticResult, list[str]]] = []
    for item in candidates:
        name = item.raw_name.casefold()
        unit = str(item.unit or "").casefold().replace(" ", "")
        signal = str(item.signal_type or "").casefold()
        score, evidence = float(item.confidence or 0.0) * 10.0, []
        if item.confirmed_label and item.confirmed_equipment_id:
            score += 100; evidence.append("human_confirmed_relation")
        if role == "heat_source_power":
            if item.physical_quantity == "power": score += 35; evidence.append("physical_quantity_power")
            if signal == "instantaneous" or "瞬時" in item.raw_name: score += 35; evidence.append("instantaneous_signal")
            if unit in POWER_UNITS: score += 15; evidence.append(f"power_unit={unit}")
            if item.physical_quantity == "energy" or signal == "cumulative" or unit in ENERGY_UNITS or any(term in name for term in ("日算", "月算", "年算", "累計", "積算", "energy")):
                score -= 120; evidence.append("cumulative_energy_excluded")
            if any(term in name for term in ("pump", "ポンプ", "fan", "aux", "補機", "heater", "ヒーター", "control", "指令")):
                score -= 45; evidence.append("auxiliary_signal_penalty")
        ranked.append((score, item, evidence))
    ranked.sort(key=lambda entry: entry[0], reverse=True)
    audit = {
        "role": role,
        "candidates": [
            {"point_id": item.point_id, "raw_name": item.raw_name, "score": round(score, 3), "evidence": evidence}
            for score, item, evidence in ranked
        ],
    }
    if not ranked:
        audit.update(selection_confidence=0.0, selection_reason="no_candidates")
        return None, audit
    best_score, best, best_evidence = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else float("-inf")
    # A confirmed relation or a material engineering score gap is sufficient.
    if len(ranked) == 1 or best.confirmed_equipment_id or best_score - second_score >= 15:
        confidence = 1.0 if best.confirmed_equipment_id else min(0.98, max(0.55, 0.65 + (best_score - second_score) / 100))
        audit.update(selected_candidate=best.point_id or best.raw_name, selection_confidence=round(confidence, 3), selection_reason=", ".join(best_evidence) or "semantic_confidence")
        return best, audit
    audit.update(selection_confidence=0.0, selection_reason="true_ambiguity_after_engineering_resolution")
    return None, audit
