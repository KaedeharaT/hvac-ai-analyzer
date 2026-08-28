from building_ai.models import SemanticResult
from building_ai.storage import Database, ProjectStore


def test_project_and_review_roundtrip(tmp_path):
    store = ProjectStore(Database(tmp_path / "test.sqlite3"))
    project = store.create("Demo", "Building A")
    assert store.get(project.project_id).building_name == "Building A"
    store.save_semantics(project.project_id, [
        SemanticResult("CH-1_LWT", "heat_source_supply_temp")
    ])
    point_id = store.load_semantics(project.project_id)[0].point_id
    store.save_review(project.project_id, point_id, "heat_source_return_temp", "checked")
    loaded = store.load_semantics(project.project_id)[0]
    assert loaded.canonical_label == "heat_source_supply_temp"
    assert loaded.effective_label == "heat_source_return_temp"
    assert loaded.human_note == "checked"


def test_same_raw_name_across_sheet_and_source_is_distinct(tmp_path):
    store = ProjectStore(Database(tmp_path / "test.sqlite3"))
    project = store.create("Demo")
    items = [
        SemanticResult("AHP-1 電力", "heat_source_power", source_file="a.xlsx", sheet="A"),
        SemanticResult("AHP-1 電力", "heat_source_power", source_file="a.xlsx", sheet="B"),
        SemanticResult("AHP-1 電力", "heat_source_power", source_file="b.xlsx", sheet="A"),
    ]
    store.save_semantics(project.project_id, items)
    loaded = store.load_semantics(project.project_id)
    assert len(loaded) == 3
    assert len({item.point_id for item in loaded}) == 3
    reviewed = next(item for item in loaded if item.source_file == "a.xlsx" and item.sheet == "B")
    store.save_review(project.project_id, reviewed.point_id, "other", "sheet-specific")
    reloaded = {item.point_id: item for item in store.load_semantics(project.project_id)}
    assert reloaded[reviewed.point_id].human_label == "other"
    assert sum(item.human_verified for item in reloaded.values()) == 1


def test_project_crud(tmp_path):
    store = ProjectStore(Database(tmp_path / "test.sqlite3"))
    project = store.create("Old")
    store.rename(project.project_id, "New")
    assert store.get(project.project_id).name == "New"
    store.delete(project.project_id)
    assert store.get(project.project_id) is None
