"""Safe discovery of locally served Qwen models.

Discovery is deliberately read-only: it never downloads a model, starts a service,
or stores a machine-specific path.  A selected result is persisted only through the
normal per-user :class:`Settings` file when the user saves Settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import requests

from building_ai.config import Settings


DEFAULT_OLLAMA_URL = "http://localhost:11434"


@dataclass(frozen=True, slots=True)
class DetectedLocalModel:
    """A usable Qwen model exposed by a local runtime."""

    model: str
    endpoint: str
    backend: str = "ollama"
    source: str = "ollama"


def _unique_urls(urls: Iterable[str]) -> list[str]:
    result: list[str] = []
    for url in urls:
        normalized = (url or "").strip().rstrip("/")
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def discover_local_qwen(settings: Settings | None = None, *, timeout: float = 1.5) -> list[DetectedLocalModel]:
    """Return Qwen models served by the local Ollama instance, if any.

    Existing product settings are checked first, followed by the standard local
    Ollama endpoint used by the research baseline.  The function is intentionally
    side-effect free, which keeps a fresh clone usable without a local LLM.
    """
    settings = settings or Settings.load()
    urls = _unique_urls((settings.ollama_url, DEFAULT_OLLAMA_URL))
    found: list[DetectedLocalModel] = []
    for url in urls:
        try:
            response = requests.get(f"{url}/api/tags", timeout=timeout)
            response.raise_for_status()
            models = response.json().get("models", [])
        except (requests.RequestException, ValueError, AttributeError):
            continue
        for item in models:
            name = str(item.get("name") or item.get("model") or "").strip()
            if name and "qwen" in name.casefold():
                candidate = DetectedLocalModel(model=name, endpoint=url)
                # The configured endpoint and default endpoint can address the
                # same local service.  A model should appear once in the UI.
                if not any(item.model.casefold() == name.casefold() for item in found):
                    found.append(candidate)
    return found


def apply_detected_local_qwen(settings: Settings, detected: DetectedLocalModel) -> None:
    """Apply a discovery result in memory; saving remains an explicit user action."""
    settings.provider = "local_qwen"
    settings.model = detected.model
    settings.ollama_url = detected.endpoint
    settings.__post_init__()
