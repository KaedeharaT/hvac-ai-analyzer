"""Factory and status facade for configured LLM providers."""

from __future__ import annotations

from building_ai.config import Settings

from .base import BaseLLMProvider, LLMUnavailableError
from .openai_provider import OpenAICompatibleProvider
from .qwen_local import LocalQwenProvider


class UnconfiguredProvider(BaseLLMProvider):
    provider_id = "not_configured"
    display_name = "Not configured"

    @property
    def is_configured(self) -> bool:
        return False

    def generate(self, prompt: str, **kwargs: object) -> str:
        raise LLMUnavailableError("LLM unavailable / not configured")

    def test_connection(self, timeout: float = 3.0) -> tuple[bool, str]:
        return False, "LLM unavailable / not configured"


class LLMManager:
    """Creates providers on demand so settings changes take effect immediately."""

    def __init__(self, settings: Settings | None = None, timeout: float = 120):
        self.settings = settings or Settings.load()
        self.timeout = timeout

    def get_provider(self) -> BaseLLMProvider:
        if self.settings.provider == "local_qwen":
            return LocalQwenProvider(self.settings, self.timeout)
        if self.settings.provider == "openai_compatible":
            return OpenAICompatibleProvider(self.settings, self.timeout)
        if self.settings.provider == "custom":
            return OpenAICompatibleProvider(self.settings, self.timeout, custom=True)
        return UnconfiguredProvider()

    def status(self, timeout: float = 1.5) -> tuple[bool, str]:
        return self.get_provider().test_connection(timeout)
