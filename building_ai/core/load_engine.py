from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class LoadResult:
    available: bool
    reason: str
    series: pd.Series | None = None


def calculate_load_ratio(load: pd.Series | None, capacity: float | pd.Series | None) -> LoadResult:
    if load is None or capacity is None:
        return LoadResult(False, "Cooling load or capacity is unavailable")
    cap = pd.to_numeric(capacity, errors="coerce") if isinstance(capacity, pd.Series) else capacity
    ratio = pd.to_numeric(load, errors="coerce") / cap
    ratio = ratio.where((ratio >= 0) & (ratio <= 2))
    return LoadResult(bool(ratio.notna().any()), "Available" if ratio.notna().any() else
                      "No physically valid load ratio values", ratio)
