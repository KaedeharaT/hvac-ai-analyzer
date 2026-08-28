import pandas as pd
import pytest

from building_ai.models import SemanticResult
from building_ai.services.agent_service import AgentService
from building_ai.storage import Database, ProjectStore, TimeseriesStore


def test_agent_blocks_unverified_abstain(tmp_path):
    projects = ProjectStore(Database(tmp_path / "db.sqlite3"))
    timeseries = TimeseriesStore(tmp_path / "ts")
    project = projects.create("Demo")
    projects.save_semantics(project.project_id, [SemanticResult("mystery", "other", abstained=True)])
    point_id = projects.load_semantics(project.project_id)[0].point_id
    timeseries.save(project.project_id, pd.DataFrame({"mystery": [1, 2]}))
    agent = AgentService(projects, timeseries)
    result = agent.tools.call("get_point_timeseries", project_id=project.project_id, point_id=point_id)
    assert not result.ok
    assert "ABSTAIN" in result.error


def test_agent_mapping_disambiguates_same_name_by_point_id(tmp_path):
    projects = ProjectStore(Database(tmp_path / "db.sqlite3"))
    timeseries = TimeseriesStore(tmp_path / "ts")
    project = projects.create("Demo")
    projects.save_semantics(project.project_id, [
        SemanticResult("Power", "terminal_power", source_file="a.xlsx", sheet="AHU"),
        SemanticResult("Power", "heat_source_power", source_file="a.xlsx", sheet="Plant"),
    ])
    mapping = AgentService(projects, timeseries).get_semantic_mapping(project.project_id)
    assert len(mapping) == 2
    assert len({item["point_id"] for item in mapping}) == 2
    assert {item["sheet"] for item in mapping} == {"AHU", "Plant"}
