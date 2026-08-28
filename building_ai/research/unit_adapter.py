"""Extract units produced by the C6 research path without product inference."""

from __future__ import annotations

import pandas as pd

from .hvac_power_col_memory import llm_score_all_slots


def extract_research_unit_db(dataframe: pd.DataFrame) -> dict[str, dict]:
    """Read C6 using the same cached slot response generated during research analysis."""
    neighbors = list(dataframe.columns)
    units: dict[str, dict] = {}
    for column in dataframe.columns:
        slots = llm_score_all_slots(column, dataframe[column], neighbor_cols=neighbors)
        c6 = slots.get("C6", {}) if isinstance(slots, dict) else {}
        units[str(column)] = {
            "unit": c6.get("unit"),
            "unit_type": c6.get("unit_type") or "unknown",
            "confidence": float(c6.get("confidence") or 0.0),
            "source": "research_c6",
        }
    return units
