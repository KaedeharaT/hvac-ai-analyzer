"""Conservative point-to-equipment organization for product analytics."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import NAMESPACE_URL, uuid5

from building_ai.core.equipment_identity import normalize_equipment_id
from building_ai.models import Equipment, EquipmentType, SemanticResult
from .power_candidate_resolver import resolve_role


HEAT_SOURCE_ROLES = {
    "heat_source_supply_temp", "heat_source_return_temp", "heat_source_flow",
    "heat_source_power", "heat_source_energy", "heat_source_capacity",
}


@dataclass(slots=True)
class HeatSourceBinding:
    equipment: Equipment
    points_by_role: dict[str, SemanticResult] = field(default_factory=dict)
    status: str = "incomplete"  # ready / incomplete / ambiguous
    reason: str = ""
    candidate_resolution: dict[str, dict] = field(default_factory=dict)

    @property
    def equipment_id(self) -> str:
        return self.equipment.equipment_id

    def point_ids(self) -> list[str]:
        return [item.point_id for item in self.points_by_role.values() if item.point_id]


@dataclass(slots=True)
class EquipmentOrganization:
    equipment: list[Equipment]
    heat_sources: list[HeatSourceBinding]


class EquipmentService:
    """Build only evidence-backed heat-source groups; ambiguity is preserved."""

    def organize(self, project_id: str, results: list[SemanticResult]) -> EquipmentOrganization:
        candidates = [
            item for item in results
            if item.effective_label in HEAT_SOURCE_ROLES and item.status.value != "ABSTAIN"
        ]
        grouped: dict[str, list[SemanticResult]] = {}
        unidentified: list[SemanticResult] = []
        for item in candidates:
            key = item.effective_equipment_id or self._equipment_key(item.raw_name)
            if key:
                grouped.setdefault(key, []).append(item)
            else:
                unidentified.append(item)

        # A single unlabelled set is still useful for small one-chiller exports.
        if unidentified and not grouped:
            grouped["heat_source"] = unidentified
        elif unidentified:
            grouped["ambiguous_unidentified"] = unidentified

        bindings = [self._binding(project_id, key, members) for key, members in grouped.items()]
        return EquipmentOrganization(
            equipment=[binding.equipment for binding in bindings], heat_sources=bindings
        )

    @staticmethod
    def _equipment_key(raw_name: str) -> str | None:
        return normalize_equipment_id(raw_name)

    def _binding(
        self, project_id: str, key: str, members: list[SemanticResult]
    ) -> HeatSourceBinding:
        name = "Heat Source" if key == "heat_source" else key
        equipment = Equipment(
            project_id=project_id,
            name=name,
            equipment_type=EquipmentType.HEAT_SOURCE,
            equipment_id=str(uuid5(NAMESPACE_URL, f"building-ai:{project_id}:{key}")),
        )
        role_candidates: dict[str, list[SemanticResult]] = {}
        for item in members:
            role_candidates.setdefault(item.effective_label, []).append(item)
        unresolved: set[str] = set()
        points: dict[str, SemanticResult] = {}
        resolutions: dict[str, dict] = {}
        for role, candidates in role_candidates.items():
            selected, audit = resolve_role(role, candidates)
            resolutions[role] = audit
            if selected is None:
                unresolved.add(role)
            else:
                points[role] = selected

        required = {
            "heat_source_supply_temp", "heat_source_return_temp",
            "heat_source_flow", "heat_source_power",
        }
        if key == "ambiguous_unidentified" or unresolved:
            reason = "multiple_candidates:" + ", ".join(sorted(unresolved or {"unidentified_equipment"}))
            return HeatSourceBinding(equipment, points, "ambiguous", reason, resolutions)
        missing = sorted(required - set(points))
        if missing:
            return HeatSourceBinding(equipment, points, "incomplete", "missing_signals:" + ", ".join(missing), resolutions)
        return HeatSourceBinding(equipment, points, "ready", "", resolutions)


def build_equipment(project_id: str, results: list[SemanticResult]) -> list[Equipment]:
    """Compatibility entry point retained for the current Equipment page."""
    return EquipmentService().organize(project_id, results).equipment
