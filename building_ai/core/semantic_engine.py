from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from building_ai.core.physics_validation import validate
from building_ai.core.unit_engine import infer_unit


@dataclass(slots=True)
class EnginePrediction:
    label: str
    confidence: float
    equipment_type: str | None
    physical_quantity: str | None
    position: str | None
    reason: str
    suspicious: bool = False


class ConservativeSemanticEngine:
    """Deprecated V1 fallback retained for external import compatibility.

    New product inference is ``core.research_semantics`` via SemanticService.
    Research source migration is documented in docs/research_to_product.md.
    """

    def predict(self, name: str, series: pd.Series) -> EnginePrediction:
        text = str(name)
        low = text.lower()
        terminal = bool(re.search(r"ahu|fcu|fan|室内機|送風|air", low, re.I))
        heat_source = bool(re.search(
            r"(?:^|[^a-z])(?:ch|hp|ahp)[-_ ]?\d|chiller|heat.?source|冷凍機|"
            r"熱源|热源|冷凍水|冷冻水|冷水|温水", low, re.I
        ))
        supply = bool(re.search(r"supply|outlet|lwt|往水|出水|送水|吹出", low, re.I))
        returns = bool(re.search(r"return|inlet|ewt|還水|回水|入口|吸込", low, re.I))
        temp = bool(re.search(r"temp|温度|水温|℃|°c|lwt|ewt", low, re.I))
        energy = bool(re.search(r"energy|電力量|积算|積算|kwh", low, re.I))
        power = not energy and bool(re.search(r"power|電力|功率|\bkw\b", low, re.I))
        flow = bool(re.search(r"flow|流量|m3/h|m³/h|l/min", low, re.I))
        capacity = bool(re.search(r"capacity|能力|容量|冷量", low, re.I))

        equipment_known = terminal ^ heat_source
        prefix = "terminal" if terminal else "heat_source"
        equipment = "Terminal" if terminal else ("Heat Source" if heat_source else None)
        if temp and (supply or returns) and equipment_known:
            position = "supply" if supply else "return"
            label = (
                f"terminal_{position}_air_temp" if terminal
                else f"heat_source_{position}_temp"
            )
            return EnginePrediction(label, 0.82, equipment, "temperature", position,
                                    "explicit equipment/quantity/position tokens")
        if flow and equipment_known:
            label = "terminal_air_volume" if terminal else "heat_source_flow"
            return EnginePrediction(label, 0.75, equipment, "flow", None,
                                    "explicit flow token")
        # Capacity often carries a kW unit; its explicit capacity token must win over power.
        for match, quantity in ((energy, "energy"), (capacity, "capacity"), (power, "power")):
            if match and equipment_known:
                return EnginePrediction(f"{prefix}_{quantity}", 0.75, equipment,
                                        quantity, None, f"explicit {quantity} token")
        known_quantity = next((quantity for matched, quantity in (
            (temp, "temperature"), (flow, "flow"), (energy, "energy"),
            (power, "power"), (capacity, "capacity"),
        ) if matched), None)
        if known_quantity:
            return EnginePrediction(
                "other", 0.65, None, known_quantity, None,
                "physical quantity found but equipment type is ambiguous",
                suspicious=True,
            )
        return EnginePrediction("other", 0.0, None, None, None,
                                "insufficient semantic evidence", suspicious=True)

    def enrich_validation(self, prediction: EnginePrediction, series: pd.Series):
        return validate(prediction.label, series)
