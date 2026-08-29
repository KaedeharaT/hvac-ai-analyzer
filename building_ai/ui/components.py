"""Reusable presentation-only widgets for the BuildingAI design system."""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from building_ai.ui.theme import SPACING_MD, SPACING_SM


class StatusBadge(QLabel):
    def __init__(self, text: str = "", tone: str = "info"):
        super().__init__(text); self.setObjectName("StatusBadge"); self.setProperty("tone", tone)
        self.setAlignment(Qt.AlignCenter); self.setMinimumHeight(24)

    def set_status(self, text: str, tone: str) -> None:
        self.setText(text); self.setProperty("tone", tone); self.style().unpolish(self); self.style().polish(self)


class SectionHeader(QWidget):
    def __init__(self, title: str, subtitle: str = "", action: QPushButton | None = None):
        super().__init__(); layout = QHBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        words = QVBoxLayout(); words.setSpacing(2); self.title = QLabel(title); self.title.setObjectName("SectionTitle"); words.addWidget(self.title)
        self.subtitle = QLabel(subtitle); self.subtitle.setObjectName("Muted"); self.subtitle.setWordWrap(True); words.addWidget(self.subtitle)
        layout.addLayout(words, 1)
        if action: layout.addWidget(action)


class EmptyState(QFrame):
    def __init__(self, title: str, description: str, action: QPushButton | None = None):
        super().__init__(); self.setObjectName("EmptyState"); layout = QVBoxLayout(self); layout.setContentsMargins(SPACING_MD, SPACING_MD, SPACING_MD, SPACING_MD)
        layout.setAlignment(Qt.AlignCenter); self.title = QLabel(title); self.title.setObjectName("EmptyStateTitle"); self.title.setAlignment(Qt.AlignCenter)
        body = QLabel(description); body.setObjectName("Muted"); body.setAlignment(Qt.AlignCenter); body.setWordWrap(True)
        layout.addWidget(self.title); layout.addWidget(body)
        if action: layout.addWidget(action, alignment=Qt.AlignCenter)
