from __future__ import annotations

import pandas as pd


def validate(label: str, series: pd.Series) -> tuple[bool | None, list[str]]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None, ["no_numeric_values"]
    warnings: list[str] = []
    if label.endswith("_temp") and ((numeric < -80) | (numeric > 150)).mean() > 0.05:
        warnings.append("temperature_out_of_broad_hvac_range")
    if label.endswith(("_flow", "_power", "_energy", "_capacity", "_volume")):
        if (numeric < 0).mean() > 0.05:
            warnings.append("unexpected_negative_values")
    return not warnings, warnings
