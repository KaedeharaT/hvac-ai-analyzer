"""Deterministic, evidence-first BEMS diagnostics for the V1 heat-source path."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import NAMESPACE_URL, uuid5

import pandas as pd

from building_ai.models import AnalysisEvidence, DiagnosticFinding
from building_ai.services.analytics_service import AnalyticsResult, EquipmentKPI


@dataclass(slots=True)
class DiagnosisConfig:
    low_cop_threshold: float = 2.5
    low_delta_t_threshold_c: float = 3.0
    workday_start_hour: int | None = None
    workday_end_hour: int | None = None
    low_load_threshold: float = 0.30
    min_samples: int = 3
    min_starts_per_day: float = 3.0


@dataclass(slots=True)
class RuleResult:
    findings: list[DiagnosticFinding] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def run_rules(analytics: AnalyticsResult, config: DiagnosisConfig | None = None) -> RuleResult:
    config = config or DiagnosisConfig()
    result = RuleResult()
    for kpi in analytics.equipment_kpis:
        if kpi.status != "available":
            result.skipped.append(f"{kpi.equipment_name}: {kpi.reason}")
    usable = analytics.available_kpis
    for kpi in usable:
        for rule in (_low_cop, _low_delta_t, _offhour, _low_load_high_power):
            finding, skipped = rule(kpi, config)
            if finding:
                result.findings.append(finding)
            if skipped:
                result.skipped.append(f"{kpi.equipment_name}: {skipped}")
    staging, skipped = _staging(usable, config)
    result.findings.extend(staging)
    result.skipped.extend(skipped)
    result.skipped.extend(analytics.skipped)
    return result


def _finding(
    kpi: EquipmentKPI, finding_type: str, severity: str, title: str, description: str,
    evidence: list[AnalysisEvidence], confidence: float, metrics: list[str], mask: pd.Series | None = None,
) -> DiagnosticFinding:
    start, end = _period(kpi.timestamps, mask)
    occurrence_count = int(mask.sum()) if mask is not None else 0
    duration_hours = _duration_hours(kpi.timestamps, mask)
    finding_id = str(uuid5(
        NAMESPACE_URL, f"building-ai:{finding_type}:{kpi.equipment_id}:{start}:{end}"
    ))
    return DiagnosticFinding(
        finding_id, kpi.equipment_id, finding_type, severity, title, description,
        evidence, confidence, {"start": start, "end": end}, metrics,
        valid_sample_count=kpi.valid_sample_count, occurrence_count=occurrence_count,
        duration_hours=duration_hours,
    )


def _period(timestamps: pd.Series | None, mask: pd.Series | None) -> tuple[str | None, str | None]:
    if timestamps is None or not timestamps.notna().any():
        return None, None
    selected = timestamps if mask is None else timestamps[mask.reindex(timestamps.index, fill_value=False)]
    selected = selected.dropna()
    return (
        selected.min().isoformat() if not selected.empty else None,
        selected.max().isoformat() if not selected.empty else None,
    )


def _duration_hours(timestamps: pd.Series | None, mask: pd.Series | None) -> float | None:
    if timestamps is None or mask is None:
        return None
    selected = timestamps[mask.reindex(timestamps.index, fill_value=False)].dropna()
    if selected.empty:
        return 0.0
    interval = _sample_interval_hours(timestamps)
    return float(len(selected) * interval) if interval > 0 else None


def _base_evidence(kpi: EquipmentKPI, name: str, value: dict, unit: str, threshold: object, note: str) -> AnalysisEvidence:
    source = kpi.evidence[0]
    return AnalysisEvidence(
        source.point_ids, kpi.equipment_id, source.start_time, source.end_time,
        name, value, unit, threshold, source.calculation_method,
        source.source_columns, [note],
    )


def _low_cop(kpi: EquipmentKPI, config: DiagnosisConfig) -> tuple[DiagnosticFinding | None, str | None]:
    cop = kpi.cop.series if kpi.cop else None
    if cop is None or cop.notna().sum() < config.min_samples:
        return None, "COP diagnosis skipped: insufficient valid COP samples"
    valid = cop.dropna()
    low = cop < config.low_cop_threshold
    low_count = int(low.sum())
    low_ratio = float(low_count / valid.size)
    if low_count < config.min_samples or low_ratio < 0.20:
        return None, None
    median = float(valid.median())
    severity = "critical" if median < config.low_cop_threshold * 0.7 or low_ratio >= 0.60 else "warning"
    evidence = _base_evidence(
        kpi, "low_cop_distribution",
        {"mean": float(valid.mean()), "median": median, "p10": float(valid.quantile(.1)),
         "low_cop_samples": low_count, "low_cop_ratio": low_ratio, "valid_samples": int(valid.size)},
        "ratio", config.low_cop_threshold,
        "V1 default threshold; site-specific commissioning may require a different threshold.",
    )
    return _finding(
        kpi, "low_heat_source_cop", severity, "Heat-source COP is persistently low",
        "The water-side COP is below the configured V1 threshold for a material portion of valid operating samples.",
        [evidence, *kpi.evidence], min(0.95, 0.55 + min(valid.size, 100) / 250),
        ["heat_source_cop", "cooling_load_kw", "input_power_kw"], low,
    ), None


def _low_delta_t(kpi: EquipmentKPI, config: DiagnosisConfig) -> tuple[DiagnosticFinding | None, str | None]:
    delta = kpi.cop.delta_t_c if kpi.cop else None
    if delta is None:
        return None, "Delta-T diagnosis skipped: temperature KPI unavailable"
    # Use the exact COP-valid mask used by the KPI cards and charts.  This
    # prevents a UI mean from silently including non-running samples.
    valid_mask = kpi.valid_mask if kpi.valid_mask is not None else (kpi.operating & (delta > 0) if kpi.operating is not None else delta > 0)
    valid = delta.where(valid_mask).dropna()
    if valid.size < config.min_samples:
        return None, "Delta-T diagnosis skipped: insufficient cooling-mode samples"
    low = (delta < config.low_delta_t_threshold_c) & valid_mask
    ratio = float(low.sum() / valid.size)
    if int(low.sum()) < config.min_samples or ratio < 0.20:
        return None, None
    evidence = _base_evidence(
        kpi, "low_delta_t_distribution",
        {"mean": float(valid.mean()), "median": float(valid.median()),
         "low_delta_t_samples": int(low.sum()), "low_delta_t_ratio": ratio,
         "valid_samples": int(valid.size)},
        "°C", config.low_delta_t_threshold_c,
        "Evaluated only while the equipment has positive power and a positive cooling temperature difference.",
    )
    severity = "critical" if float(valid.median()) < config.low_delta_t_threshold_c * .6 else "warning"
    return _finding(
        kpi, "low_chilled_water_delta_t", severity, "Chilled-water ΔT is persistently low",
        "Measured return-minus-supply temperature difference is low during equipment operation, consistent with a large-flow/small-ΔT condition.",
        [evidence, *kpi.evidence], min(0.92, 0.50 + min(valid.size, 100) / 240),
        ["chilled_water_delta_t", "heat_source_flow", "input_power_kw"], low,
    ), None


def _offhour(kpi: EquipmentKPI, config: DiagnosisConfig) -> tuple[DiagnosticFinding | None, str | None]:
    if kpi.timestamps is None or not kpi.timestamps.notna().any():
        return None, "Off-hour diagnosis skipped: imported time column is unavailable"
    # An occupancy schedule is a project setting, not an engineering default.
    # Without it, an "off-hour" finding would be an unsupported conclusion.
    if config.workday_start_hour is None or config.workday_end_hour is None:
        return None, "Off-hour diagnosis skipped: project work schedule is unavailable"
    if kpi.operating is None:
        return None, "Off-hour diagnosis skipped: power KPI unavailable"
    timestamps = kpi.timestamps
    non_work = (timestamps.dt.hour < config.workday_start_hour) | (timestamps.dt.hour >= config.workday_end_hour)
    active = kpi.operating & non_work
    active_count = int(active.sum())
    total = int(kpi.operating.sum())
    if active_count < config.min_samples or total == 0:
        return None, None
    ratio = active_count / total
    interval_hours = _sample_interval_hours(timestamps)
    power = kpi.cop.input_power_kw if kpi.cop else None
    off_hour_energy = None
    if power is not None:
        off_hour_energy = float(power.where(active, 0).fillna(0).sum() * interval_hours)
    evidence = _base_evidence(
        kpi, "off_hour_operation",
        {"off_hour_operating_samples": active_count, "operating_samples": total,
         "off_hour_operating_ratio": ratio, "off_hour_operating_hours": active_count * interval_hours,
         "off_hour_energy_kwh": off_hour_energy,
         "schedule": f"{config.workday_start_hour:02d}:00-{config.workday_end_hour:02d}:00"},
        "kWh / hours", None,
        "V1 uses a configurable default schedule; project-specific occupancy and holiday exceptions are not yet modelled.",
    )
    return _finding(
        kpi, "off_hour_operation", "warning", "Heat source operates outside configured work hours",
        "Positive equipment power was observed outside the configured V1 work-hour schedule.",
        [evidence], min(0.85, 0.45 + min(active_count, 100) / 250), ["operating_state", "input_power_kw"], active,
    ), None


def _sample_interval_hours(timestamps: pd.Series) -> float:
    intervals = timestamps.sort_values().diff().dt.total_seconds().div(3600).dropna()
    value = intervals.median() if not intervals.empty else 0.0
    return float(value) if pd.notna(value) and value > 0 else 0.0


def _low_load_high_power(kpi: EquipmentKPI, config: DiagnosisConfig) -> tuple[DiagnosticFinding | None, str | None]:
    if kpi.load_ratio is None:
        return None, "Low-load diagnosis skipped: thermal-load KPI unavailable"
    if kpi.load_mode != "rated_capacity":
        return None, "Low-load diagnosis skipped: rated capacity is unavailable"
    power = kpi.cop.input_power_kw if kpi.cop else None
    if power is None:
        return None, "Low-load diagnosis skipped: input power KPI unavailable"
    high_power = power >= power.quantile(.75)
    mask = (kpi.load_ratio < config.low_load_threshold) & high_power
    count = int(mask.sum())
    if count < config.min_samples:
        return None, None
    valid = kpi.load_ratio.dropna()
    evidence = _base_evidence(
        kpi, "low_load_high_power",
        {"samples": count, "load_ratio_median": float(valid.median()) if not valid.empty else None,
         "power_high_threshold_kw": float(power.quantile(.75)), "load_mode": kpi.load_mode},
        "ratio", config.low_load_threshold,
        "relative_load_p90 is a conservative proxy, not a rated-capacity load ratio.",
    )
    return _finding(
        kpi, "low_load_high_power", "warning", "High power observed at low thermal load",
        "The equipment operated with low calculated load while input power was in its upper operating range.",
        [evidence, *kpi.evidence], 0.72 if kpi.load_mode == "rated_capacity" else 0.58,
        ["load_ratio", "input_power_kw", "heat_source_cop"], mask,
    ), None


def _staging(kpis: list[EquipmentKPI], config: DiagnosisConfig) -> tuple[list[DiagnosticFinding], list[str]]:
    if not kpis:
        return [], ["Staging diagnosis skipped: no available heat-source KPIs"]
    findings: list[DiagnosticFinding] = []
    for kpi in kpis:
        if kpi.operating is None:
            continue
        starts = kpi.operating & ~kpi.operating.shift(1, fill_value=False)
        run_lengths = _run_lengths(kpi.operating)
        minimum_run = min(run_lengths) if run_lengths else 0
        duration_days = 0.0
        if kpi.timestamps is not None and kpi.timestamps.notna().any():
            span = (kpi.timestamps.max() - kpi.timestamps.min()).total_seconds() / 86400
            duration_days = max(span, 1 / 24)
        starts_per_day = int(starts.sum()) / duration_days if duration_days else 0.0
        if starts_per_day >= config.min_starts_per_day:
            evidence = _base_evidence(
                kpi, "start_stop_count", {"starts": int(starts.sum()), "starts_per_day": starts_per_day, "minimum_run_samples": minimum_run}, "starts", config.min_starts_per_day,
                "State is inferred from positive measured input power; start frequency is normalised by the observed period.",
            )
            findings.append(_finding(
                kpi, "frequent_start_stop", "warning", "Frequent heat-source start/stop activity",
                "Multiple power-on transitions were detected. Minimum run/off duration should be checked against site control requirements.",
                [evidence], 0.55, ["operating_state", "input_power_kw"], starts,
            ))
    if len(kpis) < 2:
        return findings, ["Parallel-operation diagnosis skipped: fewer than two heat-source equipment bindings"]
    if any(item.load_ratio is None or item.load_mode != "rated_capacity" for item in kpis):
        return findings, ["Parallel-operation diagnosis skipped: rated plant-load evidence is unavailable"]
    states = pd.concat({item.equipment_id: item.operating for item in kpis if item.operating is not None}, axis=1)
    parallel = states.sum(axis=1) >= 2
    relative_plant_load = pd.concat({item.equipment_id: item.load_ratio for item in kpis}, axis=1).mean(axis=1)
    low_load_parallel = parallel & (relative_plant_load < config.low_load_threshold)
    if int(low_load_parallel.sum()) >= config.min_samples:
        anchor = kpis[0]
        evidence = _base_evidence(
            anchor, "parallel_heat_source_operation",
            {"parallel_samples": int(parallel.sum()), "low_load_parallel_samples": int(low_load_parallel.sum()),
             "relative_plant_load_median": float(relative_plant_load[low_load_parallel].median())},
            "samples", config.low_load_threshold,
            "Plant load is the mean of available equipment load ratios; staging setpoints require site confirmation.",
        )
        findings.append(_finding(
            anchor, "parallel_heat_source_operation", "warning", "Multiple heat sources run simultaneously",
            "More than one heat-source binding had positive power in the same BEMS samples. Review staging logic with total plant load.",
            [evidence], 0.50, ["operating_state", "load_ratio"], low_load_parallel,
        ))
    return findings, []


def _run_lengths(state: pd.Series) -> list[int]:
    lengths: list[int] = []
    current = 0
    for value in state.fillna(False):
        if bool(value):
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths
