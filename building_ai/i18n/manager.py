"""Small signal-based runtime translation layer for the PyQt desktop UI."""

from __future__ import annotations

from PyQt5.QtCore import QObject, pyqtSignal

from .en_US import TRANSLATIONS as EN_US
from .zh_CN import TRANSLATIONS as ZH_CN


class LanguageManager(QObject):
    language_changed = pyqtSignal(str)
    _instance: "LanguageManager | None" = None

    def __init__(self, language: str = "en_US"):
        super().__init__()
        self.language = language if language in {"en_US", "zh_CN"} else "en_US"

    @classmethod
    def instance(cls) -> "LanguageManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_language(self, language: str) -> None:
        normalized = "zh_CN" if language in {"zh", "zh_CN"} else "en_US"
        if normalized != self.language:
            self.language = normalized
            self.language_changed.emit(normalized)

    def translate(self, key: str, **values: object) -> str:
        table = ZH_CN if self.language == "zh_CN" else EN_US
        text = table.get(key, EN_US.get(key, key))
        return text.format(**values) if values else text


def tr(key: str, **values: object) -> str:
    return LanguageManager.instance().translate(key, **values)
