from __future__ import annotations

import re

from building_ai.models import Equipment, EquipmentType, SemanticResult


def build_equipment(project_id: str, results: list[SemanticResult]) -> list[Equipment]:
    found: dict[tuple[EquipmentType, str], Equipment] = {}
    for item in results:
        raw = item.raw_name
        patterns = (
            (EquipmentType.AHU, r"\bAHU[-_ ]?\d+\b"),
            (EquipmentType.FCU, r"\bFCU[-_ ]?\d+\b"),
            (EquipmentType.CHILLER, r"\b(?:CH|CHILLER)[-_ ]?\d+\b"),
            (EquipmentType.HEAT_PUMP, r"\b(?:HP|AHP)[-_ ]?\d+\b"),
        )
        eq_type, name = EquipmentType.UNKNOWN, "Unknown"
        for candidate, pattern in patterns:
            match = re.search(pattern, raw, re.I)
            if match:
                eq_type, name = candidate, match.group(0)
                break
        if eq_type is EquipmentType.UNKNOWN and item.canonical_label.startswith("heat_source"):
            eq_type, name = EquipmentType.HEAT_SOURCE, "Heat Source"
        found.setdefault((eq_type, name), Equipment(project_id, name, eq_type))
    return list(found.values())
