from building_ai.models import SemanticResult
from building_ai.services.equipment_service import EquipmentService


def _point(name, role, number):
    return SemanticResult(name, role, point_id=f"p-{number}", unit="°C" if "temp" in role else "kW")


def test_equipment_service_creates_single_ready_heat_source():
    points = [
        _point("CH-1 supply °C", "heat_source_supply_temp", 1),
        _point("CH-1 return °C", "heat_source_return_temp", 2),
        _point("CH-1 flow m3/h", "heat_source_flow", 3),
        _point("CH-1 power kW", "heat_source_power", 4),
    ]
    result = EquipmentService().organize("project", points)
    assert len(result.heat_sources) == 1
    assert result.heat_sources[0].status == "ready"
    assert result.heat_sources[0].equipment.name == "CH-1"


def test_equipment_service_groups_multiple_numbered_heat_sources():
    points = []
    for number, prefix in enumerate(("CH-1", "CH-2"), start=1):
        for index, role in enumerate(("heat_source_supply_temp", "heat_source_return_temp", "heat_source_flow", "heat_source_power")):
            points.append(_point(f"{prefix} {role}", role, number * 10 + index))
    result = EquipmentService().organize("project", points)
    assert {item.equipment.name for item in result.heat_sources} == {"CH-1", "CH-2"}
    assert all(item.status == "ready" for item in result.heat_sources)


def test_equipment_service_marks_duplicate_role_ambiguous():
    points = [
        _point("CH-1 supply A", "heat_source_supply_temp", 1),
        _point("CH-1 supply B", "heat_source_supply_temp", 2),
        _point("CH-1 return", "heat_source_return_temp", 3),
        _point("CH-1 flow", "heat_source_flow", 4),
        _point("CH-1 power", "heat_source_power", 5),
    ]
    result = EquipmentService().organize("project", points)
    assert result.heat_sources[0].status == "ambiguous"
