"""Small signal-based runtime translation layer for the PyQt desktop UI."""

from __future__ import annotations

from PyQt5.QtCore import QObject, pyqtSignal, QCoreApplication, QLibraryInfo, QLocale, QTranslator
from PyQt5 import sip

from .en_US import TRANSLATIONS as EN_US
from .zh_CN import TRANSLATIONS as ZH_CN
from .ja_JP import TRANSLATIONS as JA_JP

TABLES = {"zh_CN": ZH_CN, "en_US": EN_US, "ja_JP": JA_JP}
LANGUAGES = (("zh_CN", "简体中文"), ("en_US", "English"), ("ja_JP", "日本語"))


def normalize_language(language: str) -> str:
    language = {"zh": "zh_CN", "en": "en_US", "ja": "ja_JP"}.get(language, language)
    return language if language in TABLES else "en_US"


class LanguageManager(QObject):
    language_changed = pyqtSignal(str)
    _instance: "LanguageManager | None" = None

    def __init__(self, language: str = "en_US"):
        super().__init__()
        self.language = normalize_language(language)
        self._qt_translator = None

    @classmethod
    def instance(cls) -> "LanguageManager":
        if cls._instance is None or sip.isdeleted(cls._instance):
            cls._instance = cls()
        return cls._instance

    def set_language(self, language: str) -> None:
        normalized = normalize_language(language)
        app = QCoreApplication.instance()
        if app is not None:
            if self._qt_translator is not None and not sip.isdeleted(self._qt_translator):
                app.removeTranslator(self._qt_translator)
            self._qt_translator = QTranslator(self)
            self._qt_translator.load('qtbase_' + normalized.split('_')[0], QLibraryInfo.location(QLibraryInfo.TranslationsPath))
            app.installTranslator(self._qt_translator)
            QLocale.setDefault(QLocale(normalized))
        if normalized != self.language:
            self.language = normalized
            self.language_changed.emit(normalized)

    def translate(self, key: str, **values: object) -> str:
        table = TABLES[self.language]
        text = table.get(key, EN_US.get(key, key))
        return text.format(**values) if values else text


def tr(key: str, **values: object) -> str:
    return LanguageManager.instance().translate(key, **values)
