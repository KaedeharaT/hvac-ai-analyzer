"""Reusable presentation-only widgets for the BuildingAI design system."""
from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from building_ai.i18n import tr
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


class GlobalContextBar(QFrame):
    """Shared project/equipment/period scope for cross-page investigation."""

    equipment_changed = pyqtSignal(object)
    period_changed = pyqtSignal(str)
    ask_ai = pyqtSignal()

    PERIODS = ("all", "24h", "7d", "30d")

    def __init__(self):
        super().__init__(); self.setObjectName("ContextBar")
        layout = QHBoxLayout(self); layout.setContentsMargins(SPACING_MD, 6, SPACING_MD, 6); layout.setSpacing(SPACING_SM)
        self.scope_label = QLabel(); self.scope_label.setObjectName("ContextLabel")
        self.project = QLabel(); self.project.setObjectName("ContextValue"); self.project.setMaximumWidth(210)
        self.equipment = QComboBox(); self.equipment.setMinimumWidth(170)
        self.period = QComboBox(); self.period.setMinimumWidth(130)
        self.readiness = StatusBadge()
        self.ai_button = QPushButton(); self.ai_button.setObjectName("PrimaryButton")
        layout.addWidget(self.scope_label); layout.addWidget(self.project); layout.addSpacing(SPACING_MD)
        layout.addWidget(self.equipment); layout.addWidget(self.period); layout.addStretch(1)
        layout.addWidget(self.readiness); layout.addWidget(self.ai_button)
        self.equipment.currentIndexChanged.connect(lambda _: self.equipment_changed.emit(self.equipment.currentData()))
        self.period.currentIndexChanged.connect(lambda _: self.period_changed.emit(self.period.currentData() or "all"))
        self.ai_button.clicked.connect(self.ask_ai)
        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self.scope_label.setText(tr("global_context"))
        selected = self.period.currentData() or "all"
        self.period.blockSignals(True); self.period.clear()
        for value in self.PERIODS:
            self.period.addItem(tr("energy_range_" + value), value)
        self.period.setCurrentIndex(max(0, self.period.findData(selected)))
        self.period.blockSignals(False)
        self.ai_button.setText(tr("ask_ai"))

    def update_context(self, context) -> None:
        project = context.current_project
        self.project.setText(project.name if project else tr("no_project"))
        selected = context.selected_equipment_id
        self.equipment.blockSignals(True); self.equipment.clear()
        self.equipment.addItem(tr("analysis_all_equipment"), None)
        for item in context.equipment:
            self.equipment.addItem(item.name, item.name)
        self.equipment.setCurrentIndex(max(0, self.equipment.findData(selected)))
        self.equipment.blockSignals(False)
        self.period.blockSignals(True); self.period.setCurrentIndex(max(0, self.period.findData(context.selected_period))); self.period.blockSignals(False)
        ready = bool(project and context.dataframe is not None and context.semantic_result is not None)
        self.readiness.set_status(tr("analysis_ready") if ready else tr("analysis_not_ready"), "success" if ready else "neutral")
