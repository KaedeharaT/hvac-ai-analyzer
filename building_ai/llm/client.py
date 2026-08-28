"""Backward-compatible facade over the pluggable provider manager.

New business code should depend on ``LLMManager`` / ``BaseLLMProvider``.  This
facade preserves the research adapters and existing public imports.
"""

from __future__ import annotations

from typing import Any

from building_ai.config import Settings

from .base import LLMError
from .manager import LLMManager


class LLMClient:
    def __init__(self, settings: Settings | None = None, timeout: float = 120):
        self.settings = settings or Settings.load()
        self.manager = LLMManager(self.settings, timeout)

    def is_available(self, timeout: float = 1.5) -> bool:
        connected, _ = self.manager.status(timeout)
        return connected

    def chat_text(self, prompt: str, system_msg: str = "", temperature: float = 0, seed: int = 0, json_mode: bool = False) -> str:
        return self.manager.get_provider().generate(
            prompt, system_prompt=system_msg, temperature=temperature, seed=seed, json_mode=json_mode,
        )

    def chat_json(self, prompt: str, system_msg: str = "Return valid JSON.", **kwargs: Any) -> dict:
        return self.manager.get_provider().generate_json(prompt, system_prompt=system_msg, **kwargs)


_default = LLMClient(Settings.from_env())


def chat_text(prompt: str, **kwargs: Any) -> str:
    return _default.chat_text(prompt, **kwargs)


def chat_json(prompt: str, system_msg: str = "Return valid JSON.", **kwargs: Any) -> dict:
    return _default.chat_json(prompt, system_msg, **kwargs)
