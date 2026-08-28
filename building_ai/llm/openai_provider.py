"""OpenAI-compatible provider for cloud and self-hosted APIs."""

from __future__ import annotations

from typing import Any

import requests

from building_ai.config import Settings

from .base import BaseLLMProvider, LLMError, LLMUnavailableError


class OpenAICompatibleProvider(BaseLLMProvider):
    provider_id = "openai_compatible"
    display_name = "OpenAI-compatible API"

    def __init__(self, settings: Settings, timeout: float = 120, custom: bool = False):
        self.settings = settings
        self.timeout = timeout
        self.custom = custom
        if custom:
            self.provider_id = "custom"
            self.display_name = "Custom / Other"

    @property
    def is_configured(self) -> bool:
        return bool(self.settings.api_base.strip() and self.settings.model.strip())

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.api_key}"} if self.settings.api_key else {}

    def test_connection(self, timeout: float = 3.0) -> tuple[bool, str]:
        if not self.is_configured:
            return False, "LLM unavailable / not configured"
        try:
            response = requests.get(f"{self.settings.api_base}/models", headers=self._headers(), timeout=timeout)
            if response.ok:
                return True, "Connected to OpenAI-compatible API"
            return False, f"API returned HTTP {response.status_code}"
        except requests.RequestException as exc:
            return False, f"Unable to reach API: {exc}"

    def generate(self, prompt: str, *, system_prompt: str = "", temperature: float = 0, seed: int = 0, json_mode: bool = False) -> str:
        if not self.is_configured:
            raise LLMUnavailableError("LLM unavailable / not configured")
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [
                *([{"role": "system", "content": system_prompt}] if system_prompt else []),
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }
        if seed:
            payload["seed"] = seed
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            response = requests.post(
                f"{self.settings.api_base}/chat/completions", json=payload,
                headers=self._headers(), timeout=self.timeout,
            )
            response.raise_for_status()
            return str(response.json()["choices"][0]["message"]["content"])
        except (requests.RequestException, KeyError, ValueError) as exc:
            raise LLMError(f"OpenAI-compatible request failed: {exc}") from exc
