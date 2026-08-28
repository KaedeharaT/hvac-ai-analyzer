from building_ai.config import Settings
from building_ai.llm import LLMClient, LLMManager


def test_unconfigured_default_keeps_application_usable(monkeypatch, tmp_path):
    for name in ("LLM_PROVIDER", "LLM_MODEL", "OLLAMA_BASE_URL", "LLM_API_BASE", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BUILDING_AI_DATA_DIR", str(tmp_path))
    settings = Settings.from_env()
    assert settings.provider == "not_configured"
    assert settings.model == ""
    assert settings.ollama_url == "http://localhost:11434"
    assert isinstance(LLMClient(settings).is_available(timeout=0.01), bool)
    provider = LLMManager(settings).get_provider()
    assert not provider.is_configured
    assert provider.test_connection()[1] == "LLM unavailable / not configured"


def test_openai_compatible_settings_save_and_reload(monkeypatch, tmp_path):
    monkeypatch.setenv("BUILDING_AI_CONFIG_DIR", str(tmp_path / "config"))
    settings = Settings(provider="openai_compatible", model="demo-model", api_base="http://localhost:8000/v1", api_key="not-a-real-key")
    settings.save()
    loaded = Settings.load()
    assert loaded.provider == "openai_compatible"
    assert loaded.model == "demo-model"
    assert loaded.api_base == "http://localhost:8000/v1"
    assert loaded.api_key == "not-a-real-key"
