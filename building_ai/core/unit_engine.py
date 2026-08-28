from __future__ import annotations

import re

import pandas as pd


def infer_unit(name: str, series: pd.Series | None = None) -> tuple[str | None, float]:
    text = str(name).lower().replace("³", "3")
    rules = (
        (r"°c|℃|degc|摂氏|摄氏", "°C"), (r"\bm3/h\b|m3/h|m³/h", "m³/h"),
        (r"\bl/min\b", "L/min"), (r"\bl/s\b", "L/s"),
        (r"\bmwh\b", "MWh"), (r"\bkwh\b", "kWh"), (r"\bmw\b", "MW"),
        (r"\bkw\b", "kW"), (r"\bwh\b", "Wh"), (r"\bw\b", "W"),
        (r"\bkpa\b", "kPa"), (r"\bpa\b", "Pa"), (r"\b%|％", "%"),
    )
    for pattern, unit in rules:
        if re.search(pattern, text, re.IGNORECASE):
            return unit, 0.95
    return None, 0.0
