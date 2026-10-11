"""Make Windows CJK fonts available to Qt, including its offscreen backend."""
import os
from pathlib import Path
from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import QApplication


def configure_fonts():
    app = QApplication.instance()
    if app is None or app.property('buildingai_fonts_loaded'): return
    directory = Path(os.environ.get('WINDIR', '')) / 'Fonts'
    if os.name == 'nt':
        for name in ('segoeui.ttf', 'msyh.ttc', 'YuGothR.ttc', 'meiryo.ttc'):
            path = directory / name
            if path.is_file(): QFontDatabase.addApplicationFont(str(path))
    families = QFontDatabase().families()
    family = next((f for f in ('Yu Gothic', 'Meiryo', 'Microsoft YaHei', 'Noto Sans CJK JP', 'Segoe UI') if f in families), app.font().family())
    app.setFont(QFont(family, 10))
    app.setProperty('buildingai_fonts_loaded', True)
