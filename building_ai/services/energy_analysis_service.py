"""Data-driven energy analysis shared by dashboards, charts, diagnosis and AI tools.

It consumes an imported frame and semantic evidence once; no project, equipment,
schedule, interval, or point-name convention is assumed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from building_ai.core.equipment_identity import normalize_equipment_id
from building_ai.models import AnalysisResult
from building_ai.services.analytics_service import AnalyticsResult
from building_ai.services.equipment_service import EquipmentOrganization
from building_ai.services.capability_detector import AnalysisCapabilities, AnalysisCapabilityDetector, CapabilityStatus


@dataclass(slots=True)
class EnergySeries:
    point_id: str | None
    name: str
    equipment_id: str | None
    equipment_name: str | None
    quantity: str
    unit: str | None
    values: pd.Series
    energy_kwh: pd.Series | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EnergyAnalysisResult:
    project_id: str
    start: str | None
    end: str | None
    sampling_interval_minutes: float | None
    data_quality: dict[str, Any]
    capabilities: dict[str, bool]
    summary: dict[str, Any]
    capability_details: dict[str, dict[str, Any]] = field(default_factory=dict)
    energy_series: list[EnergySeries] = field(default_factory=list)
    power_series: list[EnergySeries] = field(default_factory=list)
    temperature_series: list[EnergySeries] = field(default_factory=list)
    charts: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    aggregation: str = "1h"
    aggregation_rule: str | None = None
    equipment_filter: str | None = None

    def to_dict(self, *, include_values: bool = False, limit: int = 500) -> dict[str, Any]:
        def serialise(item: EnergySeries) -> dict[str, Any]:
            data = {key: value for key, value in asdict(item).items() if key not in {"values", "energy_kwh"}}
            if include_values:
                data["values"] = [None if pd.isna(v) else float(v) for v in item.values.tail(limit)]
                if item.energy_kwh is not None:
                    data["energy_kwh"] = [None if pd.isna(v) else float(v) for v in item.energy_kwh.tail(limit)]
            return data
        return {"project_id": self.project_id, "start": self.start, "end": self.end,
                "sampling_interval_minutes": self.sampling_interval_minutes,
                "data_quality": self.data_quality, "capabilities": self.capabilities,
                "capability_details": self.capability_details,
                "summary": self.summary, "charts": self.charts, "warnings": self.warnings,
                "aggregation": self.aggregation, "aggregation_rule": self.aggregation_rule,
                "equipment_filter": self.equipment_filter,
                "energy_points": [serialise(x) for x in self.energy_series],
                "power_points": [serialise(x) for x in self.power_series],
                "temperature_points": [serialise(x) for x in self.temperature_series]}


class EnergyAnalysisService:
    """Conservative calculations: unknown units or long gaps never become energy."""
    POWER_FACTORS = {"w": .001, "kw": 1., "mw": 1000.}
    ENERGY_FACTORS = {"wh": .001, "kwh": 1., "mwh": 1000.}
    LEGACY_CAPABILITIES = {
        "energy_timeseries": "energy_consumption", "power_timeseries": "power_trend",
        "temperature_timeseries": "temperature_trend", "daily_load_profile": "daily_profile",
        "load_heatmap": "heatmap", "equipment_breakdown": "equipment_ranking",
    }
    AGGREGATION_RULES = {
        "1min": "1min", "10min": "10min", "1h": "h", "1d": "D",
        "1w": "W-MON", "1mo": "MS", "1y": "YS",
    }
    RESOLUTION_MINUTES = {
        "1min": 1.0, "10min": 10.0, "1h": 60.0, "1d": 1440.0,
        "1w": 10080.0, "1mo": 40320.0, "1y": 525600.0,
    }

    def analyze(self, dataframe: pd.DataFrame, semantics: AnalysisResult, project_id: str,
                import_metadata: dict[str, Any] | None = None,
                organization: EquipmentOrganization | None = None,
                analytics: AnalyticsResult | None = None, equipment_filter: str | None = None,
                aggregation: str | None = None, *, period_start: Any | None = None,
                period_end: Any | None = None,
                comparison_periods: dict[str, tuple[Any, Any]] | None = None) -> EnergyAnalysisResult:
        timestamps = self._timestamps(dataframe, import_metadata or {})
        full_interval = self._interval_minutes(timestamps)
        scope_start, scope_end = self._normalise_period(timestamps, period_start, period_end)
        scope_mask = self._period_mask(timestamps, scope_start, scope_end)
        scope_timestamps = timestamps.loc[scope_mask].reset_index(drop=True) if timestamps is not None else None
        scope_frame = dataframe.loc[scope_mask.to_numpy()].reset_index(drop=True) if timestamps is not None else dataframe
        interval = self._interval_minutes(scope_timestamps)
        quality = self._quality(scope_frame, scope_timestamps, interval)
        no_capability = {x: False for x in self._capability_names()}
        if timestamps is None or scope_timestamps is None or scope_timestamps.empty:
            return EnergyAnalysisResult(project_id, None, None, None, quality, no_capability,
                self._summary([], [], [], organization), {}, warnings=["timestamp_unavailable"],
                aggregation=aggregation or "1h", equipment_filter=equipment_filter)
        names = {item.equipment_id: item.name for item in (organization.equipment if organization else [])}
        energy: list[EnergySeries] = []; power: list[EnergySeries] = []; temperatures: list[EnergySeries] = []; rejected_warnings: list[str] = []
        for point in semantics.semantic_results:
            column = point.column or point.raw_name
            if column not in dataframe.columns or point.status.value == "ABSTAIN": continue
            quantity = point.physical_quantity or self._quantity_from_label(point.effective_label)
            values = pd.to_numeric(dataframe[column], errors="coerce")
            if quantity == "energy" and (values < 0).any(): quality["warnings"].append(f"negative_energy:{point.raw_name}")
            if quantity == "power" and (values < 0).any(): quality["warnings"].append(f"negative_power:{point.raw_name}")
            if quantity == "temperature" and ((values < -90) | (values > 90)).any(): quality["warnings"].append(f"impossible_temperature:{point.raw_name}")
            numeric = values.dropna()
            if quantity in {"energy", "power", "temperature"} and len(numeric) >= 8:
                q1, q3 = numeric.quantile(.25), numeric.quantile(.75); iqr = q3 - q1
                if iqr > 0 and ((numeric < q1 - 4 * iqr) | (numeric > q3 + 4 * iqr)).any(): quality["warnings"].append(f"outlier_values:{point.raw_name}")
            owner = normalize_equipment_id(point.effective_equipment_id)
            if equipment_filter and owner != equipment_filter: continue
            item = EnergySeries(point.point_id, self._series_name(point, names.get(owner, owner)), owner, names.get(owner, owner), quantity or "unknown", point.unit, pd.Series(values.to_numpy(), index=timestamps))
            if quantity == "power":
                factor = self.POWER_FACTORS.get(self._unit(point.unit))
                if factor is None: item.warnings.append("unknown_or_unsupported_power_unit"); rejected_warnings.extend(item.warnings)
                else:
                    item.values *= factor; item.unit = "kW"; item.energy_kwh, warnings = self._integrate_power(item.values, timestamps, full_interval); item.warnings.extend(warnings); power.append(item)
            elif quantity == "energy":
                factor = self.ENERGY_FACTORS.get(self._unit(point.unit))
                if factor is None: item.warnings.append("unknown_or_unsupported_energy_unit"); rejected_warnings.extend(item.warnings)
                else:
                    item.values *= factor; item.unit = "kWh"; item.energy_kwh, warnings = self._energy_increments(item.values); item.warnings.extend(warnings); energy.append(item)
            elif quantity == "temperature": temperatures.append(item)
        # Convert cumulative meters before slicing. Otherwise a custom range
        # would lose its first interval or accidentally sum cumulative values.
        all_power = power
        energy = [self._scope_item(item, scope_start, scope_end) for item in energy]
        power = [self._scope_item(item, scope_start, scope_end) for item in power]
        temperatures = [self._scope_item(item, scope_start, scope_end) for item in temperatures]
        # Do not add an unscoped power series to an unscoped direct meter: the
        # source hierarchy is unknowable, so summing would double-count.
        direct_owners = {x.equipment_id for x in energy}
        energy_for_charts = [*energy, *[x for x in power if x.equipment_id not in direct_owners]]
        aggregation_name, rule = self._resolve_aggregation(aggregation, scope_timestamps, full_interval)
        charts = self._charts(scope_timestamps, energy_for_charts, power, temperatures, analytics,
                              equipment_filter, rule, comparison_periods, all_power)
        details = AnalysisCapabilityDetector().detect(
            charts=charts, timestamp_available=True, energy_points=len(energy_for_charts),
            power_points=len(power), temperature_points=len(temperatures),
            analytics_available=bool(analytics and analytics.available_kpis), quality=quality,
        )
        if rule in {"D", "W-MON", "MS", "YS"} and power:
            details.statuses["daily_profile"] = CapabilityStatus(False, "requires_subdaily_resolution")
            details.statuses["heatmap"] = CapabilityStatus(False, "requires_subdaily_resolution")
        capabilities = details.boolean_flags()
        capabilities.update({legacy: capabilities[current] for legacy, current in self.LEGACY_CAPABILITIES.items()})
        return EnergyAnalysisResult(project_id, scope_timestamps.min().isoformat(), scope_timestamps.max().isoformat(), interval, quality, capabilities,
            self._summary(energy_for_charts, power, temperatures, organization, analytics, scope_timestamps, equipment_filter), details.to_dict(), energy, power, temperatures, charts,
            [*rejected_warnings, *[warning for item in [*energy, *power, *temperatures] for warning in item.warnings]],
            aggregation_name, rule, equipment_filter)

    @staticmethod
    def _normalise_period(timestamps: pd.Series | None, start: Any | None, end: Any | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        if timestamps is None or timestamps.dropna().empty:
            return None, None
        lower = pd.to_datetime(start, errors="coerce") if start is not None else timestamps.min()
        upper = pd.to_datetime(end, errors="coerce") if end is not None else timestamps.max()
        lower = timestamps.min() if pd.isna(lower) else pd.Timestamp(lower)
        upper = timestamps.max() if pd.isna(upper) else pd.Timestamp(upper)
        if lower > upper:
            raise ValueError("period_start must not be after period_end")
        return lower, upper

    @staticmethod
    def _period_mask(timestamps: pd.Series | None, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.Series:
        if timestamps is None:
            return pd.Series(dtype=bool)
        return timestamps.notna() & (timestamps >= start) & (timestamps <= end)

    @staticmethod
    def _scope_item(item: EnergySeries, start: pd.Timestamp | None, end: pd.Timestamp | None) -> EnergySeries:
        mask = item.values.index.notna() & (item.values.index >= start) & (item.values.index <= end)
        return EnergySeries(item.point_id, item.name, item.equipment_id, item.equipment_name,
                            item.quantity, item.unit, item.values.loc[mask],
                            item.energy_kwh.loc[mask] if item.energy_kwh is not None else None,
                            list(item.warnings))

    def _resolve_aggregation(self, aggregation: str | None, timestamps: pd.Series,
                             native_interval_minutes: float | None = None) -> tuple[str, str]:
        name = aggregation
        if name is None:
            name = next((candidate for candidate in self.AGGREGATION_RULES
                         if self.resolution_supported(candidate, native_interval_minutes)), "1y")
        if name not in self.AGGREGATION_RULES:
            raise ValueError(f"Unsupported energy aggregation: {name}")
        if not self.resolution_supported(name, native_interval_minutes):
            raise ValueError(
                f"Requested resolution {name} is finer than the native sampling interval "
                f"({native_interval_minutes:g} min); upsampling is not permitted"
            )
        return name, self.AGGREGATION_RULES[name]

    @classmethod
    def resolution_supported(cls, aggregation: str, native_interval_minutes: float | None) -> bool:
        """Return whether aggregation can be formed without inventing samples."""
        if aggregation not in cls.RESOLUTION_MINUTES or native_interval_minutes is None:
            return aggregation in cls.RESOLUTION_MINUTES
        requested = cls.RESOLUTION_MINUTES[aggregation]
        # Calendar buckets have variable duration.  These lower bounds keep a
        # native monthly/yearly series selectable without allowing it to be
        # presented as a finer weekly/monthly series.
        if aggregation == "1mo":
            return native_interval_minutes <= 31 * 1440
        if aggregation == "1y":
            return native_interval_minutes <= 366 * 1440
        return requested + 1e-9 >= native_interval_minutes

    @staticmethod
    def _capability_names() -> tuple[str, ...]:
        return ("energy_consumption", "power_trend", "temperature_trend", "delta_t", "thermal_load", "cop", "daily_profile", "heatmap", "weather_correlation", "equipment_ranking", "period_comparison", "energy_timeseries", "power_timeseries", "temperature_timeseries", "daily_load_profile", "load_heatmap", "equipment_breakdown")

    @staticmethod
    def _unit(unit: str | None) -> str | None:
        return str(unit).strip().replace("³", "3").replace("℃", "°C").lower() if unit else None

    @staticmethod
    def _quantity_from_label(label: str) -> str | None:
        return "temperature" if label.endswith("_temp") else "flow" if label.endswith("_flow") else "power" if label.endswith("_power") else "energy" if label.endswith("_energy") else None

    @staticmethod
    def _series_name(point, equipment_name: str | None) -> str:
        """Present a mapped engineering role, not an opaque semantic ID."""
        roles = {
            "heat_source_supply_temp": "Supply water temperature",
            "heat_source_return_temp": "Return water temperature",
            "terminal_supply_air_temp": "Supply air temperature",
            "terminal_return_air_temp": "Return air temperature",
            "outdoor_air_temp": "Outdoor air temperature",
            "heat_source_power": "Input power",
            "heat_source_energy": "Energy consumption",
        }
        label = roles.get(point.effective_label, point.raw_name)
        return f"{equipment_name}: {label}" if equipment_name else label

    @staticmethod
    def _timestamps(df: pd.DataFrame, metadata: dict[str, Any]) -> pd.Series | None:
        name = metadata.get("time_column")
        if name in df.columns:
            values = pd.to_datetime(df[name], errors="coerce")
            if values.notna().any(): return values.reset_index(drop=True)
        return None

    @staticmethod
    def _interval_minutes(timestamps: pd.Series | None) -> float | None:
        if timestamps is None: return None
        diff = timestamps.sort_values().diff().dt.total_seconds().div(60).dropna(); value = diff.median() if not diff.empty else None
        return float(value) if value is not None and pd.notna(value) and value > 0 else None

    @staticmethod
    def _quality(df: pd.DataFrame, timestamps: pd.Series | None, interval: float | None) -> dict[str, Any]:
        result = {"missing_ratio": float(df.isna().mean().mean()) if df.size else 0., "duplicate_timestamps": 0,
                  "irregular_sampling": False, "flatline_points": [], "warnings": []}
        if timestamps is None: result["warnings"].append("timestamp_unavailable")
        else:
            valid = timestamps.dropna(); result["duplicate_timestamps"] = int(valid.duplicated().sum())
            diffs = valid.sort_values().diff().dt.total_seconds().div(60).dropna()
            if interval and not diffs.empty: result["irregular_sampling"] = bool((diffs.sub(interval).abs() > max(.01, interval * .05)).mean() > .05)
            if result["duplicate_timestamps"]: result["warnings"].append("duplicate_timestamps")
            if result["irregular_sampling"]: result["warnings"].append("irregular_sampling")
        for column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            if len(values) >= 3 and values.nunique() == 1: result["flatline_points"].append(str(column))
        return result

    @staticmethod
    def _integrate_power(power_kw: pd.Series, timestamps: pd.Series, interval: float | None) -> tuple[pd.Series, list[str]]:
        delta = timestamps.shift(-1).sub(timestamps).dt.total_seconds().div(3600); inferred = (interval or 0.) / 60
        if inferred > 0: delta = delta.fillna(inferred)
        # Timestamp values are data, not dataframe indexes.  Align elapsed
        # durations positionally to the power series before multiplication.
        delta = pd.Series(delta.to_numpy(), index=power_kw.index)
        valid = (delta > 0) & ((inferred <= 0) | (delta <= inferred * 3)); values = power_kw.where(valid).clip(lower=0) * delta.where(valid)
        return values, ["power_integration_excluded_irregular_or_missing_intervals"] if int((~valid).sum()) else []

    @staticmethod
    def _energy_increments(values: pd.Series) -> tuple[pd.Series, list[str]]:
        diff = values.diff(); observed = diff.dropna(); cumulative = bool(len(observed) >= 2 and (observed >= 0).mean() >= .95 and values.dropna().nunique() > 2)
        if cumulative:
            resets = diff < 0; warnings = ["cumulative_meter_differenced"]
            if resets.any(): warnings.append("cumulative_meter_reset_excluded")
            return diff.where(diff >= 0), warnings
        return values.where(values >= 0), ["energy_values_treated_as_interval_values"]

    @staticmethod
    def _aggregate(series: pd.Series, rule: str | None, operation: str = "sum", *, include_count: bool = False) -> list[dict[str, Any]]:
        clean = series.loc[series.index.notna()].sort_index()
        if rule is None:
            values = clean
            counts = clean.notna().astype(int)
        else:
            kwargs = {"label": "left", "closed": "left"} if rule == "W-MON" else {}
            grouped = clean.resample(rule, **kwargs)
            values = grouped.sum(min_count=1) if operation == "sum" else grouped.mean()
            counts = grouped.count()
        rows = []
        for key, value in values.dropna().items():
            row = {"time": key.isoformat(), "value": float(value)}
            if include_count:
                row["valid_sample_count"] = int(counts.get(key, 0))
            rows.append(row)
        return rows

    def _charts(self, timestamps: pd.Series, energy: list[EnergySeries], power: list[EnergySeries], temperatures: list[EnergySeries], analytics: AnalyticsResult | None, equipment_filter: str | None = None, rule: str | None = None,
                comparison_periods: dict[str, tuple[Any, Any]] | None = None,
                all_power: list[EnergySeries] | None = None) -> dict[str, Any]:
        charts: dict[str, Any] = {}
        if energy:
            total = pd.concat([x.energy_kwh.rename(x.name) for x in energy if x.energy_kwh is not None], axis=1).sum(axis=1, min_count=1)
            if total.notna().any():
                name = f"{energy[0].equipment_name}: Energy consumption" if equipment_filter and energy[0].equipment_name else "Building total energy"
                charts["energy_trend"] = {"unit": "kWh", "aggregation": rule, "aggregation_operation": "sum", "series": [{"name": name, "data": self._aggregate(total, rule)}]}
        if power:
            total = pd.concat([x.values for x in power], axis=1).sum(axis=1, min_count=1)
            if total.notna().any():
                name = f"{power[0].equipment_name}: Input power" if equipment_filter and power[0].equipment_name else "Building total power"
                charts["power_trend"] = {"unit": "kW", "aggregation": rule, "aggregation_operation": "mean", "peak_operation": "raw_max", "series": [{"name": name, "data": self._aggregate(total, rule, "mean")}], "peak_kw": float(total.max()), "peak_time": total.idxmax().isoformat(), "average_kw": float(total.mean())}
                kwargs = {"label": "left", "closed": "left"} if rule == "W-MON" else {}
                aggregated_power = total.resample(rule, **kwargs).mean()
                if rule not in {"D", "W-MON", "MS", "YS"}:
                    profile = aggregated_power.groupby([aggregated_power.index.weekday < 5, aggregated_power.index.strftime("%H:%M")]).mean()
                    charts["daily_load_profile"] = {"unit": "kW", "aggregation": rule, "aggregation_operation": "mean", "series": [{"name": "Weekday" if bool(k) else "Weekend", "data": [{"time": t, "value": float(v)} for (_, t), v in profile[profile.index.get_level_values(0) == k].items()]} for k in sorted(profile.index.get_level_values(0).unique(), reverse=True)]}
                    charts["load_heatmap"] = {"unit": "kW", "aggregation": rule, "aggregation_operation": "mean", "data": [{"date": key[0].isoformat(), "time": key[1], "value": float(value)} for key, value in aggregated_power.groupby([aggregated_power.index.date, aggregated_power.index.strftime("%H:%M")]).mean().items()]}
                comparison = self._period_comparison(all_power or power, comparison_periods)
                if comparison_periods is not None:
                    if comparison:
                        charts["period_comparison"] = comparison
                else:
                    daily = total.resample("D").mean().dropna()
                    if len(daily) >= 4:
                        half = len(daily) // 2; previous, current = float(daily.iloc[:half].mean()), float(daily.iloc[half:].mean())
                        charts["period_comparison"] = {"unit": "kW", "data": [{"name": "Period A", "value": previous}, {"name": "Period B", "value": current}], "basis": "observed_period_halves"}
        if temperatures:
            series = [{"name": x.name, "unit": x.unit or "unknown", "data": self._aggregate(x.values, rule, "mean")} for x in temperatures[:4] if x.values.notna().any()]
            if series: charts["temperature_trend"] = {"unit": "mixed/unknown", "aggregation": rule, "aggregation_operation": "mean", "series": series}
            outdoor = next((x for x in temperatures if any(t in x.name.casefold() for t in ("outdoor", "outside", "外気", "外气", "外温"))), None)
            if outdoor and power:
                total = pd.concat([x.values for x in power], axis=1).sum(axis=1, min_count=1)
                kwargs = {"label": "left", "closed": "left"} if rule == "W-MON" else {}
                outdoor_values = outdoor.values.resample(rule, **kwargs).mean()
                power_values = total.resample(rule, **kwargs).mean()
                paired = pd.DataFrame({"temperature": outdoor_values, "power": power_values}).dropna()
                if len(paired) >= 3: charts["weather_correlation"] = {"x_unit": outdoor.unit or "unknown", "y_unit": "kW", "aggregation": rule, "aggregation_operation": "mean", "sample_count": int(len(paired)), "data": [{"x": float(row.temperature), "y": float(row.power)} for row in paired.iloc[:2000].itertuples()]}
        if analytics:
            delta = []; cop = []
            for item in analytics.available_kpis:
                if equipment_filter and normalize_equipment_id(item.equipment_name) != equipment_filter: continue
                if item.timestamps is None or item.cop is None: continue
                def scoped(values: pd.Series) -> pd.Series:
                    result = pd.Series(values.to_numpy(), index=pd.to_datetime(item.timestamps, errors="coerce"))
                    if item.valid_mask is not None:
                        result = result.where(pd.Series(item.valid_mask.to_numpy(), index=result.index))
                    return result.loc[(result.index >= timestamps.min()) & (result.index <= timestamps.max())]
                if item.cop.delta_t_c is not None: delta.append({"name": item.equipment_name, "data": self._aggregate(scoped(item.cop.delta_t_c), rule, "mean")})
                if item.cop.series is not None: cop.append({"name": item.equipment_name, "data": self._aggregate(scoped(item.cop.series), rule, "mean", include_count=True)})
            if delta: charts["delta_t_trend"] = {"unit": "°C", "aggregation": rule, "aggregation_operation": "mean", "series": delta}
            if cop: charts["cop_trend"] = {"unit": "ratio", "aggregation": rule, "aggregation_operation": "mean", "series": cop}
        ranking = [{"name": x.equipment_name, "value": float(x.energy_kwh.sum())} for x in energy if x.equipment_name and x.energy_kwh is not None and pd.notna(x.energy_kwh.sum(min_count=1))]
        if len(ranking) >= 2: charts["equipment_ranking"] = {"unit": "kWh", "data": sorted(ranking, key=lambda x: x["value"], reverse=True)}
        return charts

    @staticmethod
    def _period_comparison(power: list[EnergySeries], periods: dict[str, tuple[Any, Any]] | None) -> dict[str, Any] | None:
        if not periods or not power:
            return None
        total = pd.concat([item.values for item in power], axis=1).sum(axis=1, min_count=1)
        rows = []
        bounds: dict[str, dict[str, str]] = {}
        for key, display in (("a", "Period A"), ("b", "Period B")):
            raw = periods.get(key)
            if not raw or len(raw) != 2:
                return None
            start, end = pd.to_datetime(raw[0], errors="coerce"), pd.to_datetime(raw[1], errors="coerce")
            if pd.isna(start) or pd.isna(end) or start > end:
                return None
            values = total.loc[(total.index >= start) & (total.index <= end)].dropna()
            if values.empty:
                return None
            rows.append({"name": display, "value": float(values.mean()), "valid_sample_count": int(values.count())})
            bounds[key] = {"start": pd.Timestamp(start).isoformat(), "end": pd.Timestamp(end).isoformat()}
        return {"unit": "kW", "data": rows, "basis": "custom_periods", "periods": bounds, "aggregation": "mean"}

    @staticmethod
    def _summary(energy: list[EnergySeries], power: list[EnergySeries], temperatures: list[EnergySeries], organization: EquipmentOrganization | None, analytics: AnalyticsResult | None = None, scope_timestamps: pd.Series | None = None, equipment_filter: str | None = None) -> dict[str, Any]:
        totals = [x.energy_kwh.sum(min_count=1) for x in energy if x.energy_kwh is not None]; total_energy = sum(float(x) for x in totals if pd.notna(x)) if totals else None
        merged = pd.concat([x.values for x in power], axis=1).sum(axis=1, min_count=1) if power else pd.Series(dtype=float)
        kpis = analytics.available_kpis if analytics else []
        def stats(key: str) -> tuple[float | None, int]:
            if scope_timestamps is not None and not scope_timestamps.empty:
                starts, ends = scope_timestamps.min(), scope_timestamps.max(); values: list[pd.Series] = []
                source_name = {"cop": "series", "delta_t_c": "delta_t_c"}.get(key)
                if source_name:
                    for item in kpis:
                        if equipment_filter and normalize_equipment_id(item.equipment_name) != equipment_filter:
                            continue
                        source = getattr(item.cop, source_name, None) if item.cop else None
                        if source is None or item.timestamps is None: continue
                        indexed = pd.Series(source.to_numpy(), index=pd.to_datetime(item.timestamps, errors="coerce"))
                        if item.valid_mask is not None:
                            indexed = indexed.where(pd.Series(item.valid_mask.to_numpy(), index=indexed.index))
                        valid = indexed.loc[(indexed.index >= starts) & (indexed.index <= ends)].dropna()
                        if not valid.empty: values.append(valid.reset_index(drop=True))
                    if values:
                        combined = pd.concat(values, ignore_index=True)
                        return float(combined.mean()), int(combined.count())
            summary_values = [x.metric_summary.get(key, {}) for x in kpis if not equipment_filter or normalize_equipment_id(x.equipment_name) == equipment_filter]
            weighted = [(x.get("mean"), int(x.get("valid_count") or 0)) for x in summary_values if x.get("mean") is not None]
            count = sum(item[1] for item in weighted)
            if not weighted:
                return None, 0
            if count:
                return float(sum(value * samples for value, samples in weighted) / count), count
            return float(np.mean([value for value, _ in weighted])), 0
        cop_mean, cop_count = stats("cop"); delta_mean, delta_count = stats("delta_t_c")
        return {"total_energy_kwh": total_energy, "peak_power_kw": float(merged.max()) if not merged.empty else None, "average_power_kw": float(merged.mean()) if not merged.empty else None,
                "average_cop": cop_mean, "average_cop_valid_sample_count": cop_count,
                "average_delta_t_c": delta_mean, "average_delta_t_valid_sample_count": delta_count,
                "energy_points": len(energy), "power_points": len(power), "temperature_points": len(temperatures), "equipment_count": (1 if equipment_filter else len(organization.equipment)) if organization else 0}
