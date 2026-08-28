import pandas as pd
import pytest

from building_ai.config import Settings
from building_ai.core.equipment_identity import normalize_equipment_id
from building_ai.models import AnalysisResult, SemanticResult
from building_ai.services.analytics_service import AnalyticsService
from building_ai.services.semantic_service import SemanticService
from building_ai.services.equipment_service import EquipmentService
from building_ai.storage import DuplicateImportError
from building_ai.ui.context import ApplicationContext


def _point(equipment_id: str, role: str, number: int) -> SemanticResult:
    unit = "°C" if role.endswith("temp") else "m3/h" if role.endswith("flow") else "kW"
    return SemanticResult(f"{equipment_id} {role}", role, point_id=f"p-{number}", equipment_id=equipment_id, unit=unit)


def test_equipment_identifier_normalization_is_precise():
    assert normalize_equipment_id("AHP#3-1 往水温度") == "AHP-3-1"
    assert normalize_equipment_id("AHP 3-1 流量") == "AHP-3-1"
    assert normalize_equipment_id("ＡＨＰ＃３－１ 消費電力") == "AHP-3-1"
    assert normalize_equipment_id("AHP-3-11 power") == "AHP-3-11"
    assert normalize_equipment_id("AHP-3-1 power") != normalize_equipment_id("AHP-3-11 power")


def test_equipment_groups_are_isolated_before_kpi_selection():
    roles = ("heat_source_supply_temp", "heat_source_return_temp", "heat_source_flow", "heat_source_power")
    points = [_point(f"AHP-3-{device}", role, device * 10 + index) for device in range(1, 5) for index, role in enumerate(roles)]

    organization = EquipmentService().organize("project", points)

    assert {item.equipment.name for item in organization.heat_sources} == {"AHP-3-1", "AHP-3-2", "AHP-3-3", "AHP-3-4"}
    assert all(item.status == "ready" for item in organization.heat_sources)
    assert all(len(item.points_by_role) == 4 for item in organization.heat_sources)


def test_duplicate_role_within_one_equipment_remains_ambiguous():
    roles = ("heat_source_supply_temp", "heat_source_return_temp", "heat_source_flow", "heat_source_power")
    points = [_point("AHP-3-1", role, index) for index, role in enumerate(roles)]
    points.append(_point("AHP-3-1", "heat_source_power", 99))

    binding = EquipmentService().organize("project", points).heat_sources[0]

    assert binding.status == "ambiguous"
    assert "heat_source_power" in binding.reason


def test_kpi_uses_each_equipment_group_not_project_wide_candidates():
    roles = ("heat_source_supply_temp", "heat_source_return_temp", "heat_source_flow", "heat_source_power")
    points = [_point(f"AHP-3-{device}", role, device * 10 + index) for device in (1, 2) for index, role in enumerate(roles)]
    frame = pd.DataFrame({
        item.raw_name: ([7, 7] if item.effective_label.endswith("supply_temp") else [12, 12] if item.effective_label.endswith("return_temp") else [10, 10] if item.effective_label.endswith("flow") else [10, 10])
        for item in points
    })
    result = AnalyticsService().analyze_project(frame, AnalysisResult(points), "project")

    assert len(result.equipment_kpis) == 2
    assert all(item.status == "available" for item in result.equipment_kpis)


def test_power_resolver_excludes_daily_energy_and_selects_instantaneous_power(tmp_path):
    frame = pd.DataFrame({
        "AHP#3-1 往水温度": [7.0, 7.1], "AHP#3-1 還水温度": [12.0, 12.1],
        "AHP#3-1 流量": [500, 510], "AHP#3-1 消費電力（瞬時値）": [50, 51],
        "AHP#3-1 消費電力（日算値）": [100, 150],
    })
    semantics = SemanticService(Settings(provider="not_configured", data_dir=tmp_path / "data")).analyze_dataframe(frame)
    items = {item.raw_name: item for item in semantics.semantic_results}
    assert items["AHP#3-1 消費電力（瞬時値）"].effective_label == "heat_source_power"
    assert items["AHP#3-1 消費電力（日算値）"].effective_label == "other"
    binding = EquipmentService().organize("project", semantics.semantic_results).heat_sources[0]
    assert binding.status == "ready"
    assert binding.points_by_role["heat_source_power"].raw_name == "AHP#3-1 消費電力（瞬時値）"


def test_project_import_is_managed_and_survives_restart(tmp_path):
    source = tmp_path / "source.csv"
    pd.DataFrame({"timestamp": ["2025-01-01", "2025-01-02"], "AHP#3-1 往水温度": [7.0, 7.1]}).to_csv(source, index=False)
    settings = Settings(data_dir=tmp_path / "app-data")
    context = ApplicationContext(settings)
    project = context.projects.create("Persisted")
    context.open_project(project.project_id)

    context.import_data(str(source))
    managed = context.import_metadata["imports"][0]["managed_path"]
    assert context.timeseries.exists(project.project_id)
    assert pd.io.common.file_exists(managed)
    source.unlink()

    restarted = ApplicationContext(Settings(data_dir=tmp_path / "app-data"))
    restarted.open_project(project.project_id)
    assert restarted.dataframe is not None
    assert len(restarted.dataframe) == 2
    assert restarted.import_metadata["imports"][0]["original_filename"] == "source.csv"


def test_project_add_replace_clear_and_duplicate_invalidation(tmp_path):
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    pd.DataFrame({"timestamp": ["2025-01-01"], "AHP-3-1 power kW": [10]}).to_csv(first, index=False)
    pd.DataFrame({"timestamp": ["2025-01-02"], "AHP-3-1 power kW": [20]}).to_csv(second, index=False)
    context = ApplicationContext(Settings(data_dir=tmp_path / "app-data"))
    project = context.projects.create("Lifecycle")
    context.open_project(project.project_id)
    context.import_data(str(first))
    with pytest.raises(DuplicateImportError):
        context.import_data(str(first))
    context.import_data(str(second), mode="add")
    assert len(context.dataframe) == 2
    assert len(context.import_metadata["imports"]) == 2

    context.run_semantics()
    assert context.semantic_result is not None
    context.import_data(str(second), mode="replace")
    assert len(context.dataframe) == 1
    assert context.semantic_result is None
    assert context.projects.load_semantics(project.project_id) == []
    assert context.projects.get(project.project_id).analysis_summary["status"] == "stale"

    context.clear_project_data()
    assert context.projects.get(project.project_id) is not None
    assert context.dataframe is None
    assert not context.timeseries.exists(project.project_id)
