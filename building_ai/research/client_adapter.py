"""OpenAI-shaped adapter required by untouched research helper functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from building_ai.llm.base import BaseLLMProvider
from building_ai.llm.client import LLMClient


@dataclass(slots=True)
class _Message:
    content: str


@dataclass(slots=True)
class _Choice:
    message: _Message


@dataclass(slots=True)
class _Response:
    choices: list[_Choice]


class _CompletionsAdapter:
    def __init__(self, client: LLMClient | BaseLLMProvider):
        self._client = client

    def create(
        self, model: str | None = None, messages: list[dict[str, Any]] | None = None,
        temperature: float = 0.0, max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None, **kwargs: Any,
    ) -> _Response:
        messages = messages or []
        system = "\n".join(
            str(item.get("content", "")) for item in messages
            if item.get("role") == "system"
        )
        conversation = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}"
            for item in messages if item.get("role") != "system"
        )
        if isinstance(self._client, BaseLLMProvider):
            content = self._client.generate(
                conversation, system_prompt=system, temperature=temperature,
                seed=int(kwargs.get("seed", 0) or 0),
            )
        else:
            content = self._client.chat_text(
                conversation, system_msg=system, temperature=temperature,
                seed=int(kwargs.get("seed", 0) or 0),
            )
        return _Response([_Choice(_Message(content))])


class _ChatAdapter:
    def __init__(self, client: LLMClient | BaseLLMProvider):
        self.completions = _CompletionsAdapter(client)


class ResearchLLMClientAdapter:
    """Expose ``chat.completions.create`` without changing research algorithms."""

    def __init__(self, client: LLMClient | BaseLLMProvider):
        self.chat = _ChatAdapter(client)
