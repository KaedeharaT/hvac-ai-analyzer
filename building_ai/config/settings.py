"""Application settings with environment overrides and no embedded secrets.

Runtime data stays beside the project by default, while user preferences (including an
optional API key) are stored in the operating-system user configuration directory.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIRECT_PROMPT_VERSIONS = {
    "direct_v1_name",
    "direct_v2_name_values",
    "direct_v3_name_values_stats",
    "direct_v4_full_context",
}


def user_config_dir() -> Path:
    """Return a platform-appropriate per-user directory without hard-coded paths."""
    override = os.getenv("BUILDING_AI_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        return Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming")) / "BuildingAI"
    return Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / "building-ai"


@dataclass(slots=True)
class Settings:
    # ``not_configured`` keeps all deterministic product features usable on a
    # fresh clone. Legacy names are normalized in ``__post_init__``.
    provider: str = "not_configured"
    model: str = ""
    ollama_url: str = "http://localhost:11434"
    api_base: str = ""
    api_key: str = ""
    local_model_path: str = ""
    local_device: str = "auto"
    drawing_model_path: str = ""
    language: str = "en"
    data_dir: Path = field(default_factory=lambda: user_config_dir() / "data")
    direct_prompt_version: str = "direct_v1_name"
    task_queue_backend: str = "local"
    redis_url: str = "redis://localhost:6379/0"

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        aliases = {
            "ollama": "local_llm", "local_llm": "local_llm",
            # Backward compatibility for settings saved before the provider-neutral rename.
            "qwen": "local_llm", "local_qwen": "local_llm",
            "openai": "openai_compatible", "groq": "openai_compatible",
            "openrouter": "openai_compatible",
        }
        self.provider = aliases.get(self.provider.strip().lower(), self.provider.strip().lower())
        self.ollama_url = self.ollama_url.rstrip("/")
        self.api_base = self.api_base.rstrip("/")
        self.language = "zh_CN" if self.language in {"zh", "zh_CN"} else "en_US" if self.language in {"en", "en_US"} else self.language
        self.task_queue_backend = self.task_queue_backend.strip().lower()
        if self.task_queue_backend not in {"local", "redis"}:
            raise ValueError("TASK_QUEUE_BACKEND must be 'local' or 'redis'")
        if self.direct_prompt_version not in DIRECT_PROMPT_VERSIONS:
            raise ValueError(f"Unsupported Direct Prompt version: {self.direct_prompt_version}")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            provider=os.getenv("LLM_PROVIDER", "not_configured"),
            model=os.getenv("LLM_MODEL", ""),
            ollama_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            api_base=os.getenv("LLM_API_BASE", ""),
            api_key=os.getenv("LLM_API_KEY", ""),
            local_model_path=os.getenv("LOCAL_MODEL_PATH", ""),
            local_device=os.getenv("LOCAL_LLM_DEVICE", "auto"),
            drawing_model_path=os.getenv("BUILDING_AI_DRAWING_MODEL", ""),
            language=os.getenv("BUILDING_AI_LANGUAGE", "en"),
            data_dir=Path(os.getenv("BUILDING_AI_DATA_DIR", str(user_config_dir() / "data"))),
            direct_prompt_version=os.getenv("DIRECT_PROMPT_VERSION", "direct_v1_name"),
            task_queue_backend=os.getenv("BUILDING_AI_TASK_QUEUE", "local"),
            redis_url=os.getenv("BUILDING_AI_REDIS_URL", "redis://localhost:6379/0"),
        )

    @property
    def database_path(self) -> Path:
        return self.data_dir / "database" / "building_ai.sqlite3"

    @property
    def timeseries_dir(self) -> Path:
        return self.data_dir / "projects"

    @property
    def settings_path(self) -> Path:
        return user_config_dir() / "settings.json"

    @property
    def confirmed_dataset_path(self) -> Path:
        """Product-only human-confirmed mappings; never paper ground truth."""
        return user_config_dir() / "confirmed_semantics.sqlite3"

    def save(self) -> None:
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["data_dir"] = str(self.data_dir)
        self.settings_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls) -> "Settings":
        env = cls.from_env()
        if not env.settings_path.exists():
            return env
        try:
            return cls(**json.loads(env.settings_path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return env
