"""Provider-neutral LLM contracts used by product services."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any


class LLMError(RuntimeError):
    """Raised when an enabled provider cannot complete a request."""


class LLMUnavailableError(LLMError):
    """Raised when no provider has been configured."""


class BaseLLMProvider(ABC):
    """Minimal common interface for local and OpenAI-compatible chat models."""

    provider_id = "base"
    display_name = "LLM"

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Whether this provider has enough configuration to make a request."""

    @abstractmethod
    def generate(
        self, prompt: str, *, system_prompt: str = "", temperature: float = 0,
        seed: int = 0, json_mode: bool = False,
    ) -> str:
        """Generate a text response from the configured model."""

    @abstractmethod
    def test_connection(self, timeout: float = 3.0) -> tuple[bool, str]:
        """Return a user-facing connection result without raising for normal errors."""

    def generate_json(self, prompt: str, *, system_prompt: str = "Return valid JSON.", **kwargs: Any) -> dict:
        text = self.generate(prompt, system_prompt=system_prompt, json_mode=True, **kwargs)
        cleaned = text.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMError("The provider did not return valid JSON") from exc
