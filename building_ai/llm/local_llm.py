"""Optional Ollama-backed local LLM provider."""

from __future__ import annotations

from typing import Any

import requests

from building_ai.config import Settings

from .base import BaseLLMProvider, LLMError, LLMUnavailableError


class LocalLLMProvider(BaseLLMProvider):
    provider_id = "local_llm"
    display_name = "Local open-source LLM"

    def __init__(self, settings: Settings, timeout: float = 120):
        self.settings = settings
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        return bool(self.settings.model.strip() and self.settings.ollama_url.strip())

    def test_connection(self, timeout: float = 3.0) -> tuple[bool, str]:
        if not self.is_configured:
            return False, "LLM unavailable / not configured"
        try:
            response = requests.get(f"{self.settings.ollama_url}/api/tags", timeout=timeout)
            if not response.ok:
                return False, f"Ollama returned HTTP {response.status_code}"
            models = {item.get("name") for item in response.json().get("models", [])}
            if self.settings.model not in models:
                return False, f"Ollama is reachable, but model '{self.settings.model}' is not installed"
            return True, "Connected to local LLM"
        except (requests.RequestException, ValueError) as exc:
            return False, f"Unable to reach Ollama: {exc}"

    def generate(self, prompt: str, *, system_prompt: str = "", temperature: float = 0, seed: int = 0, json_mode: bool = False) -> str:
        if not self.is_configured:
            raise LLMUnavailableError("LLM unavailable / not configured")
        payload: dict[str, Any] = {
            "model": self.settings.model, "stream": False,
            "messages": [
                *([{"role": "system", "content": system_prompt}] if system_prompt else []),
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": temperature, "seed": seed},
        }
        if json_mode:
            payload["format"] = "json"
        try:
            response = requests.post(f"{self.settings.ollama_url}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
            return str(response.json()["message"]["content"])
        except (requests.RequestException, KeyError, ValueError) as exc:
            raise LLMError(f"local LLM request failed: {exc}") from exc
