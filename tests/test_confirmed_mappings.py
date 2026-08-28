import pandas as pd

from building_ai.config import Settings
from building_ai.models import SemanticResult
from building_ai.services import SemanticService
from building_ai.services.equipment_service import EquipmentService
from building_ai.storage import ConfirmedMappingStore, Database, ProjectStore
from building_ai.ui.analysis_renderer import reason_text
from building_ai.i18n import LanguageManager


def test_confirmed_mapping_persists_with_project_review(tmp_path):
    store = ProjectStore(Database(tmp_path / "project.sqlite3"))
    project = store.create("A")
    item = SemanticResult("CH-1 Power", "heat_source_power", point_id="p1")
    store.save_semantics(project.project_id, [item])
    store.save_review(project.project_id, "p1", "terminal_power", "verified", "AHU-01")
    loaded = store.get_semantic(project.project_id, "p1")
    assert loaded.canonical_label == "heat_source_power"
    assert loaded.confirmed_label == "terminal_power"
    assert loaded.effective_label == "terminal_power"
    assert loaded.effective_equipment_id == "AHU-01"
    assert loaded.review_status == "CONFIRMED"


def test_exact_confirmed_dataset_reuse_but_not_similar_name(tmp_path, monkeypatch):
    monkeypatch.setenv("BUILDING_AI_CONFIG_DIR", str(tmp_path / "user"))
    settings = Settings(provider="not_configured")
    confirmed = ConfirmedMappingStore(settings.confirmed_dataset_path)
    confirmed.save("old-project", SemanticResult("AHU-01 Supply Air Temp", "terminal_supply_air_temp", confirmed_equipment_id="AHU-01"), "human checked")
    service = SemanticService(settings)
    exact = service.analyze_dataframe(pd.DataFrame({"AHU-01 Supply Air Temp": [18.0, 19.0]}))
    similar = service.analyze_dataframe(pd.DataFrame({"AHU-02 Supply Air Temp": [18.0, 19.0]}))
    assert exact.semantic_results[0].confidence == 1.0
    assert exact.semantic_results[0].debug_metadata["llm_prior"]["source"] == "confirmed_dataset_exact"
    assert exact.semantic_results[0].effective_equipment_id == "AHU-01"
    assert similar.semantic_results[0].debug_metadata["llm_prior"]["source"] != "confirmed_dataset_exact"


def test_equipment_grouping_uses_confirmed_effective_label():
    labels = ("heat_source_supply_temp", "heat_source_return_temp", "heat_source_flow", "heat_source_power")
    items = [SemanticResult(f"CH-1 {label}", "other", point_id=str(i), confirmed_label=label) for i, label in enumerate(labels)]
    organization = EquipmentService().organize("project", items)
    assert organization.heat_sources[0].status == "ready"


def test_analysis_reason_rerenders_without_changing_identifier():
    manager = LanguageManager.instance()
    manager.set_language("zh_CN")
    assert reason_text("multiple_candidates:heat_source_power") == "检测到多个 heat_source_power 候选点位"
    manager.set_language("en_US")
    assert reason_text("multiple_candidates:heat_source_power") == "Multiple candidates for heat_source_power"
