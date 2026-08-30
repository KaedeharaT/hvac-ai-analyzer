from building_ai.config import Settings
from building_ai.llm import apply_detected_local_model, discover_local_models


class _Response:
    def raise_for_status(self):
        return None

    @staticmethod
    def json():
        return {"models": [{"name": "local-model-a"}, {"name": "local-model-b"}, {"model": "local-model-c"}]}


def test_local_llm_discovery_returns_available_models(monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append((url, timeout))
        return _Response()

    monkeypatch.setattr("building_ai.llm.local_discovery.requests.get", fake_get)
    settings = Settings(provider="local_llm", ollama_url="http://detected-host:11434")

    found = discover_local_models(settings)

    assert [item.model for item in found] == ["local-model-a", "local-model-b", "local-model-c"]
    assert found[0].endpoint == "http://detected-host:11434"
    assert calls[0][0].endswith("/api/tags")


def test_discovered_model_can_be_applied_and_persisted(monkeypatch, tmp_path):
    monkeypatch.setenv("BUILDING_AI_CONFIG_DIR", str(tmp_path / "user-config"))
    settings = Settings()
    from building_ai.llm.local_discovery import DetectedLocalModel

    apply_detected_local_model(settings, DetectedLocalModel("local-model", "http://localhost:11434"))
    settings.save()
    restored = Settings.load()

    assert restored.provider == "local_llm"
    assert restored.model == "local-model"
    assert restored.ollama_url == "http://localhost:11434"
