import pandas as pd

from building_ai.config import Settings
from building_ai.models import SemanticResult
from building_ai.ui.context import ApplicationContext


def _seed(context, name: str, equipment: str):
    project = context.projects.create(name)
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=4, freq="h"),
        f"{equipment} Supply °C": [7.0] * 4,
        f"{equipment} Return °C": [12.0] * 4,
        f"{equipment} Flow m3/h": [3.6] * 4,
        f"{equipment} Power kW": [8.0] * 4,
    })
    context.timeseries.save(project.project_id, frame)
    context.projects.save_import_metadata(project.project_id, {"time_column": "timestamp", "start": "2025-01-01T00:00:00", "end": "2025-01-01T03:00:00", "sampling_interval": "0 days 01:00:00", "imports": [{"original_filename": f"{name}.csv"}]})
    points = [
        SemanticResult(f"{equipment} Supply °C", "heat_source_supply_temp", point_id=f"{project.project_id}-s", column=f"{equipment} Supply °C", equipment_id=equipment, physical_quantity="temperature", unit="°C"),
        SemanticResult(f"{equipment} Return °C", "heat_source_return_temp", point_id=f"{project.project_id}-r", column=f"{equipment} Return °C", equipment_id=equipment, physical_quantity="temperature", unit="°C"),
        SemanticResult(f"{equipment} Flow m3/h", "heat_source_flow", point_id=f"{project.project_id}-f", column=f"{equipment} Flow m3/h", equipment_id=equipment, physical_quantity="flow", unit="m³/h"),
        SemanticResult(f"{equipment} Power kW", "heat_source_power", point_id=f"{project.project_id}-p", column=f"{equipment} Power kW", equipment_id=equipment, physical_quantity="power", unit="kW"),
    ]
    context.projects.save_semantics(project.project_id, points)
    project.data_status = "ready"; project.analysis_summary = {"status": "current"}; project.time_range = {"start": "2025-01-01T00:00:00", "end": "2025-01-01T03:00:00"}; context.projects.save(project)
    return project, points


def test_agent_tools_restore_persisted_project_data_after_restart_and_follow_selection(tmp_path):
    settings = Settings(provider="not_configured", data_dir=tmp_path / "data")
    first = ApplicationContext(settings)
    alpha, alpha_points = _seed(first, "Alpha", "CH-01")
    beta, _ = _seed(first, "Beta", "HP-02")

    # New ApplicationContext models an application restart: no runtime cache is reused.
    restarted = ApplicationContext(settings)
    restarted.open_project(alpha.project_id)
    summary = restarted.agent.tools.call("get_project_summary", project_id=alpha.project_id).data
    assert summary["data_available"] is True
    assert summary["number_of_rows"] == 4
    assert summary["discovered_equipment"] == ["CH-01"]
    assert restarted.agent.tools.call("get_point_timeseries", project_id=alpha.project_id, point_id=alpha_points[-1].point_id).ok
    assert restarted.agent.tools.call("get_analysis_results", project_id=alpha.project_id).data["available"] is True

    restarted.open_project(beta.project_id)
    switched = restarted.agent.tools.call("get_project_summary", project_id=beta.project_id).data
    assert switched["project_name"] == "Beta"
    assert switched["discovered_equipment"] == ["HP-02"]
    assert "CH-01" not in str(restarted.agent.tools.call("get_equipment_summary", project_id=beta.project_id).data)
    # Controller routing follows the MainWindow-selected project too; it does
    # not retain Alpha in a separate Agent session cache.
    answer = restarted.agent_controller.answer("what equipment is in the current project?")
    assert "HP-02" in answer
    assert "CH-01" not in answer
