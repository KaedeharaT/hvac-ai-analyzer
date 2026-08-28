from building_ai.config import Settings
from building_ai.llm import apply_detected_local_qwen, discover_local_qwen


class _Response:
    def raise_for_status(self):
        return None

    @staticmethod
    def json():
        return {"models": [{"name": "llama3:8b"}, {"name": "qwen2.5:7b"}, {"model": "Qwen3:4b"}]}


def test_local_qwen_discovery_only_returns_qwen_models(monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append((url, timeout))
        return _Response()

    monkeypatch.setattr("building_ai.llm.local_discovery.requests.get", fake_get)
    settings = Settings(provider="local_qwen", ollama_url="http://detected-host:11434")

    found = discover_local_qwen(settings)

    assert [item.model for item in found] == ["qwen2.5:7b", "Qwen3:4b"]
    assert found[0].endpoint == "http://detected-host:11434"
    assert calls[0][0].endswith("/api/tags")


def test_discovered_model_can_be_applied_and_persisted(monkeypatch, tmp_path):
    monkeypatch.setenv("BUILDING_AI_CONFIG_DIR", str(tmp_path / "user-config"))
    settings = Settings()
    from building_ai.llm.local_discovery import DetectedLocalModel

    apply_detected_local_qwen(settings, DetectedLocalModel("qwen2.5:7b", "http://localhost:11434"))
    settings.save()
    restored = Settings.load()

    assert restored.provider == "local_qwen"
    assert restored.model == "qwen2.5:7b"
    assert restored.ollama_url == "http://localhost:11434"
