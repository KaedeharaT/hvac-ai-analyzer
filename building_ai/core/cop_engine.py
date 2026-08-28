from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(slots=True)
class CopResult:
    available: bool
    reason: str
    series: pd.Series | None = None
    summary: dict[str, float] | None = None
    cooling_load_kw: pd.Series | None = None
    delta_t_c: pd.Series | None = None
    input_power_kw: pd.Series | None = None
    valid_count: int = 0
    filtered_count: int = 0
    warnings: list[str] = field(default_factory=list)
    conversion_method: str = ""
    # Normalized signals retained for one canonical analysis mask and display.
    flow_lps: pd.Series | None = None
    supply_temp_c: pd.Series | None = None
    return_temp_c: pd.Series | None = None


def _canonical_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    return str(unit).strip().replace("³", "3").replace("℃", "°C").lower()


def _flow_to_lps(values: pd.Series, unit: str | None) -> tuple[pd.Series | None, str | None]:
    canonical = _canonical_unit(unit)
    if canonical == "l/s":
        return values, "flow L/s → L/s"
    if canonical == "l/min":
        return values / 60.0, "flow L/min ÷ 60 → L/s"
    if canonical == "m3/h":
        return values * (1000.0 / 3600.0), "flow m³/h × 1000/3600 → L/s"
    return None, None


def _power_to_kw(values: pd.Series, unit: str | None) -> tuple[pd.Series | None, str | None]:
    canonical = _canonical_unit(unit)
    if canonical == "kw":
        return values, "power kW → kW"
    if canonical == "w":
        return values / 1000.0, "power W ÷ 1000 → kW"
    return None, None


def calculate_heat_source_cop(
    df: pd.DataFrame, supply: str | None, returns: str | None,
    flow: str | None, power: str | None, flow_unit: str | None = None,
    power_unit: str | None = None, temperature_unit: str | None = "°C",
    min_cop: float = 0.5, max_cop: float = 15.0,
) -> CopResult:
    """Calculate cooling-mode water-side COP without guessing missing units."""
    missing = [name for name, value in {
        "supply temperature": supply, "return temperature": returns,
        "flow": flow, "power": power,
    }.items() if not value or value not in df.columns]
    if missing:
        return CopResult(False, "Missing " + ", ".join(missing))
    if _canonical_unit(temperature_unit) not in {"°c", "c"}:
        return CopResult(False, f"Unsupported or unknown temperature unit: {temperature_unit!r}")

    s, r, raw_flow, raw_power = (
        pd.to_numeric(df[column], errors="coerce") for column in (supply, returns, flow, power)
    )
    flow_lps, flow_method = _flow_to_lps(raw_flow, flow_unit)
    if flow_lps is None:
        return CopResult(False, f"Unsupported or unknown flow unit: {flow_unit!r}")
    power_kw, power_method = _power_to_kw(raw_power, power_unit)
    if power_kw is None:
        return CopResult(False, f"Unsupported or unknown power unit: {power_unit!r}")

    delta_t = r - s
    cooling_load = 4.186 * flow_lps * delta_t
    base_valid = (flow_lps > 0) & (power_kw > 0) & (delta_t > 0)
    raw_cop = (cooling_load / power_kw.replace(0, np.nan)).where(base_valid)
    cop = raw_cop.where((raw_cop > min_cop) & (raw_cop < max_cop))
    valid = cop.dropna()
    warnings: list[str] = []
    non_positive_dt = int((delta_t.notna() & (delta_t <= 0)).sum())
    if non_positive_dt:
        warnings.append(f"non_positive_cooling_delta_t_samples={non_positive_dt}")
    filtered = int(raw_cop.notna().sum() - valid.size)
    if valid.empty:
        return CopResult(
            False, "No physically valid cooling-mode COP values", series=cop,
            cooling_load_kw=cooling_load, delta_t_c=delta_t, input_power_kw=power_kw, filtered_count=filtered,
            warnings=warnings, conversion_method=f"{flow_method}; {power_method}; Q=4.186×flow(L/s)×ΔT(°C)",
            flow_lps=flow_lps, supply_temp_c=s, return_temp_c=r,
        )
    return CopResult(
        True, "Available", cop,
        {
            "mean": float(valid.mean()), "median": float(valid.median()),
            "min": float(valid.min()), "max": float(valid.max()),
            "p10": float(valid.quantile(0.10)), "p90": float(valid.quantile(0.90)),
            "valid_count": float(valid.count()),
        }, cooling_load, delta_t, power_kw, int(valid.size), filtered, warnings,
        f"{flow_method}; {power_method}; Q=4.186×flow(L/s)×ΔT(°C); COP=Q/power(kW)",
        flow_lps, s, r,
    )
