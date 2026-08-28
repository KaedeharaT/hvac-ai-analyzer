"""Product implementation of the current paper-research semantic boundary.

Research source: paper_research/src/bems_v3/core.py and
paper_research/src/bems_v4/core.py.  This module deliberately contains no
experiment, ground-truth, cache, project-specific, or paper-directory imports.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any

import pandas as pd

from building_ai.core.physics_validation import validate
from building_ai.core.preprocessing import normalize_header
from building_ai.core.unit_engine import infer_unit
from building_ai.core.equipment_identity import normalize_equipment_id
from building_ai.models.semantic_result import STRICT_8_TAXONOMY


NON_TARGET = re.compile(r"湿度|humidity|\brh\b|圧力|pressure|設定|setpoint|指令|command|能力|capacity|"
                        r"運転容量|\bcop\b|異常|alarm|fault|status|開度|damper|valve|熱量|steam|蒸気|"
                        r"電力量|積算|累計|日算値|月算値|年算値|daily|monthly|cumulative|\bkwh\b|\bmwh\b", re.I)


@dataclass(frozen=True, slots=True)
class PointEvidence:
    normalized_name: str
    equipment_id: str | None
    equipment_type: str
    equipment_scope: str
    loop_context: str
    quantity: str
    role: str
    medium: str
    measurement_type: str
    unit: str | None
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self); result["evidence"] = list(self.evidence); return result


def parse_point(name: str, supplied_unit: str | None = None) -> PointEvidence:
    """Extract explainable, bilingual engineering evidence from a point name."""
    text = normalize_header(name); low = text.lower(); evidence: list[str] = []
    equipment_id = normalize_equipment_id(text)
    if equipment_id: evidence.append(f"equipment_id={equipment_id}")
    if re.search(r"ahu|fcu|air handling|空調機|fan coil", low, re.I) or (equipment_id and equipment_id.startswith(("AHU-", "FCU-"))): equipment = "ahu_terminal"
    elif re.search(r"heat pump|ヒートポンプ|chiller|チラー|冷凍機|熱源|ashp|wshp|ahp", low, re.I) or (equipment_id and equipment_id.startswith(("AHP-", "ASHP-", "WSHP-", "HP-", "CH-", "CHILLER-", "CHU-"))): equipment = "heat_source"
    elif re.search(r"heat exchanger|熱交換|\bhex\b", low, re.I): equipment = "heat_exchanger"
    elif re.search(r"pump|ポンプ|\bpch\b|\bpwu\b", low, re.I): equipment = "pump"
    elif re.search(r"負荷側|ヘッダ|header|load side", low, re.I): equipment = "load_header"
    elif re.search(r"受電盤|分電盤|panel|whm", low, re.I): equipment = "electrical"
    else: equipment = "unknown"
    if re.search(r"露点|dew\s*point", low, re.I): quantity = "dew_point"
    elif re.search(r"湿度|humidity|\brh\b", low, re.I): quantity = "humidity"
    elif re.search(r"co2|二酸化炭素", low, re.I): quantity = "co2"
    elif re.search(r"圧力|差圧|pressure|静圧", low, re.I): quantity = "pressure"
    elif re.search(r"電力量|積算|累計|日算値|月算値|年算値|daily|monthly|cumulative|\bkwh\b|\bmwh\b|\bwh\b|energy", low, re.I): quantity = "energy"
    elif re.search(r"電力|消費電力|\bkw\b|\bmw\b|\bw\b|\bpower\b", low, re.I): quantity = "power"
    elif re.search(r"流量|flow|m3/h|m³/h|l/min|l/s", low, re.I): quantity = "flow"
    elif re.search(r"温度|temp|℃|°c|lwt|ewt", low, re.I): quantity = "temperature"
    elif re.search(r"設定|setpoint|\bsp\b", low, re.I): quantity = "setpoint"
    elif re.search(r"指令|command", low, re.I): quantity = "command"
    elif re.search(r"状態|運転|status|alarm|fault", low, re.I): quantity = "status"
    else: quantity = "unknown"
    if re.search(r"給気|supply\s*air|吹出|送り側|送水|往水|往温|供給|supply|outlet|lwt", low, re.I): role = "supply"
    elif re.search(r"還気|return\s*air|吸込|返り側|還水|回水|還温|return|inlet|ewt", low, re.I): role = "return"
    else: role = "unknown"
    if re.search(r"給気|還気|外気|室内|吸込|吹出|room|air|ahu|fcu|fan", low, re.I) and not re.search(r"水|water|冷温水|温水|冷水", low, re.I): medium = "air"
    elif quantity in {"power", "energy"}: medium = "electric"
    elif re.search(r"水|water|冷温水|温水|冷水|地中熱|井水|lwt|ewt", low, re.I): medium = "water"
    else: medium = "unknown"
    if equipment_id: scope = "device_level"
    elif equipment == "load_header": scope = "load_header"
    elif equipment == "electrical": scope = "building_level"
    else: scope = "unknown"
    if re.search(r"地中熱|ground", low, re.I): loop = "ground_source"
    elif re.search(r"生成|generation|source-side", low, re.I): loop = "generation"
    elif scope == "load_header": loop = "load_header"
    elif medium == "air" and equipment == "ahu_terminal": loop = "air_side"
    elif medium == "water": loop = "water_side"
    else: loop = "unknown"
    if equipment != "unknown": evidence.append(f"equipment={equipment}")
    if loop != "unknown": evidence.append(f"loop={loop}")
    unit, _ = infer_unit(text) if not supplied_unit else (supplied_unit, 1.0)
    measurement_type = "command" if re.search(r"指令|command", low, re.I) else "setpoint" if re.search(r"設定|setpoint|\bsp\b", low, re.I) else "measured"
    return PointEvidence(text, equipment_id, equipment, scope, loop, quantity, role, medium, measurement_type, unit, tuple(evidence))


def direct_prompt(name: str) -> str:
    labels = "\n".join(f"- {label}" for label in STRICT_8_TAXONOMY)
    return f"""Classify one BEMS/HVAC point name using only the strict taxonomy. Return JSON only.\ncolumn_name: {json.dumps(name, ensure_ascii=False)}\nlabels:\n{labels}\nReturn {{\"label\": \"one label\", \"confidence\": 0.0, \"reason\": \"brief reason\"}}."""


def conservative_gate(label: str, raw_name: str) -> tuple[str, str | None]:
    """Paper V3's selected physical non-target contradiction gate."""
    if label != "other" and NON_TARGET.search(raw_name): return "other", "explicit_non_target_quantity"
    return label, None


def deterministic_label(point: PointEvidence) -> tuple[str, float, str]:
    """Offline fallback: label only fully compatible, device-scoped evidence."""
    if point.measurement_type != "measured": return "other", 0.35, "explicit_non_target_control_or_setpoint"
    if point.equipment_scope in {"load_header", "building_level"}: return "other", 0.92, "explicit_non_device_scope"
    if point.equipment_type == "heat_source" and point.equipment_scope == "device_level":
        if point.quantity == "temperature" and point.medium == "water" and point.role in {"supply", "return"}: return f"heat_source_{point.role}_temp", 0.93, "device_heat_source_water_temperature_role"
        if point.quantity == "flow" and point.medium in {"water", "unknown"}: return "heat_source_flow", 0.88 if point.medium == "water" else 0.78, "device_heat_source_water_flow" if point.medium == "water" else "device_heat_source_flow_no_explicit_medium"
        if point.quantity == "power" and point.medium == "electric": return "heat_source_power", 0.88, "device_heat_source_electrical_power"
    if point.equipment_type == "ahu_terminal":
        if point.quantity == "temperature" and point.medium == "air" and point.role == "supply": return "terminal_supply_air_temp", 0.93, "ahu_air_temperature_supply"
        if point.quantity == "temperature" and point.medium == "air" and point.role == "return": return "terminal_return_air_temp", 0.93, "ahu_air_temperature_return"
        if point.quantity == "power": return "terminal_power", 0.88, "ahu_electrical_power"
    # Safe local fallback for exports that name an explicit chilled/hot-water
    # circuit but omit the individual equipment identifier.  It remains REVIEW
    # and EquipmentService will never use it as a unique ready binding.
    if point.equipment_type == "unknown" and point.medium == "water":
        if point.quantity == "temperature" and point.role in {"supply", "return"}: return f"heat_source_{point.role}_temp", 0.60, "insufficient_equipment_id_for_water_temperature_role"
        if point.quantity == "flow": return "heat_source_flow", 0.60, "insufficient_equipment_id_for_water_flow"
    return "other", 0.0 if point.quantity == "unknown" else 0.45, "insufficient_strict_device_quantity_role_evidence"


def validate_mapping(label: str, point: PointEvidence, values: pd.Series) -> tuple[bool | None, list[str]]:
    valid, warnings = validate(label, values)
    expected = "temperature" if label.endswith("_temp") else "flow" if label.endswith("_flow") else "power" if label.endswith("_power") else None
    if expected and point.quantity not in {expected, "unknown"}: warnings.append(f"quantity_incompatible:{point.quantity}")
    if label != "other" and point.measurement_type != "measured": warnings.append("control_or_setpoint_not_measurement")
    if label != "other" and point.equipment_scope in {"load_header", "building_level"}: warnings.append("non_device_scope")
    return (False if warnings else valid), list(dict.fromkeys(warnings))
