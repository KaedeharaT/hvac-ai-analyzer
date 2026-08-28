"""Single capability model for data-driven BEMS analyses.

The detector is deliberately independent of any UI.  It records both an
availability decision and the conservative reason behind that decision, so the
dashboard, Energy Analysis page, diagnosis workflow, and Agent expose the same
answer to "what can this project support?".
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


CAPABILITY_NAMES = (
    "energy_consumption", "power_trend", "temperature_trend", "delta_t",
    "thermal_load", "cop", "daily_profile", "heatmap",
    "weather_correlation", "equipment_ranking", "period_comparison",
)


@dataclass(frozen=True, slots=True)
class CapabilityStatus:
    available: bool
    reason: str


@dataclass(slots=True)
class AnalysisCapabilities:
    """Formal, serializable product capability result."""
    statuses: dict[str, CapabilityStatus] = field(default_factory=dict)

    def available(self, name: str) -> bool:
        return self.statuses.get(name, CapabilityStatus(False, "not_evaluated")).available

    def reason(self, name: str) -> str:
        return self.statuses.get(name, CapabilityStatus(False, "not_evaluated")).reason

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {name: asdict(self.statuses[name]) for name in CAPABILITY_NAMES}

    def boolean_flags(self) -> dict[str, bool]:
        return {name: self.available(name) for name in CAPABILITY_NAMES}


class AnalysisCapabilityDetector:
    """Derive supported analyses only from validated, real service outputs."""

    CHART_CAPABILITIES = {
        "energy_consumption": "energy_trend",
        "power_trend": "power_trend",
        "temperature_trend": "temperature_trend",
        "delta_t": "delta_t_trend",
        "cop": "cop_trend",
        "daily_profile": "daily_load_profile",
        "heatmap": "load_heatmap",
        "weather_correlation": "weather_correlation",
        "equipment_ranking": "equipment_ranking",
        "period_comparison": "period_comparison",
    }

    def detect(
        self, *, charts: dict[str, Any], timestamp_available: bool,
        energy_points: int, power_points: int, temperature_points: int,
        analytics_available: bool, quality: dict[str, Any],
    ) -> AnalysisCapabilities:
        statuses: dict[str, CapabilityStatus] = {}
        for capability, chart in self.CHART_CAPABILITIES.items():
            if chart in charts:
                statuses[capability] = CapabilityStatus(True, "available")
                continue
            statuses[capability] = CapabilityStatus(False, self._reason(
                capability, timestamp_available, energy_points, power_points,
                temperature_points, analytics_available, quality,
            ))
        # Thermal load is an equipment KPI rather than a standalone chart.
        statuses["thermal_load"] = CapabilityStatus(
            bool(analytics_available and charts.get("cop_trend")),
            "available" if analytics_available and charts.get("cop_trend") else "requires_supply_return_flow_and_power",
        )
        return AnalysisCapabilities(statuses)

    @staticmethod
    def _reason(name: str, timestamp_available: bool, energy_points: int,
                power_points: int, temperature_points: int,
                analytics_available: bool, quality: dict[str, Any]) -> str:
        if not timestamp_available:
            return "timestamp_unavailable"
        if quality.get("duplicate_timestamps"):
            return "duplicate_timestamps_require_resolution"
        if name == "energy_consumption":
            return "requires_energy_meter_or_supported_power_unit"
        if name in {"power_trend", "daily_profile", "heatmap", "period_comparison"}:
            return "requires_supported_power_timeseries"
        if name == "temperature_trend":
            return "requires_mapped_temperature_timeseries"
        if name in {"delta_t", "cop", "thermal_load"}:
            return "requires_supply_return_flow_and_power"
        if name == "weather_correlation":
            return "requires_outdoor_temperature_and_power"
        if name == "equipment_ranking":
            return "requires_two_or_more_equipment_energy_series"
        return "insufficient_validated_signals"
