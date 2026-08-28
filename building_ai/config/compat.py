"""Names expected by the behavior-preserving research implementation."""

from .settings import DIRECT_PROMPT_VERSIONS, Settings

_settings = Settings.from_env()
LLM_PROVIDER = _settings.provider
LLM_MODEL = _settings.model
OLLAMA_BASE_URL = _settings.ollama_url
DIRECT_PROMPT_VERSION = _settings.direct_prompt_version
PROMPT_VERSION = DIRECT_PROMPT_VERSION
