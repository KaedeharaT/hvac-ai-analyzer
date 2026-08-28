"""Reusable, presentation-only widgets for the read-only AI Assistant page."""

from __future__ import annotations

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QTextEdit, QVBoxLayout, QWidget

from building_ai.i18n import LanguageManager, tr
from building_ai.ui.theme import SPACING_SM


class ChatInput(QTextEdit):
    """Multiline editor where Enter submits and Shift+Enter inserts a newline."""

    submitted = pyqtSignal()

    def keyPressEvent(self, event):  # noqa: N802 - Qt API name
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not event.modifiers() & Qt.ShiftModifier:
            event.accept()
            self.submitted.emit()
            return
        super().keyPressEvent(event)


class ChatMessage(QFrame):
    """Selectable, wrapped chat content with an optional translation key."""

    def __init__(self, role: str, text: str, *, translation_key: str | None = None):
        super().__init__()
        self.role = role
        self.raw_text = text
        self.translation_key = translation_key
        self.setObjectName("ChatMessageUser" if role == "user" else "ChatMessageAssistant")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING_SM, SPACING_SM, SPACING_SM, SPACING_SM)
        layout.setSpacing(3)
        self.role_label = QLabel()
        self.role_label.setObjectName("ChatMessageRole")
        self.body = QLabel()
        self.body.setObjectName("ChatMessageBody")
        self.body.setWordWrap(True)
        self.body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.role_label)
        layout.addWidget(self.body)
        LanguageManager.instance().language_changed.connect(self.retranslate_ui)
        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self.role_label.setText(tr("chat_you") if self.role == "user" else tr("chat_assistant"))
        self.body.setText(tr(self.translation_key) if self.translation_key else self.raw_text)


class ToolCallWidget(QFrame):
    """A compact status row reserved for future structured Agent tool calls."""

    STATES = {"RUNNING", "SUCCESS", "FAILED"}

    def __init__(self, tool_id: str, status: str = "RUNNING", detail: str = ""):
        super().__init__()
        self.tool_id = tool_id
        self.status = status if status in self.STATES else "RUNNING"
        self.detail = detail
        self.setObjectName("ToolCall")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACING_SM, 5, SPACING_SM, 5)
        self.icon = QLabel()
        self.label = QLabel()
        self.label.setWordWrap(True)
        layout.addWidget(self.icon)
        layout.addWidget(self.label, 1)
        LanguageManager.instance().language_changed.connect(self.retranslate_ui)
        self.retranslate_ui()

    def update_status(self, status: str, detail: str = "") -> None:
        self.status = status if status in self.STATES else "RUNNING"
        if detail:
            self.detail = detail
        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        icon, key = {
            "RUNNING": ("…", "tool_running"),
            "SUCCESS": ("✓", "tool_success"),
            "FAILED": ("⚠", "tool_failed"),
        }[self.status]
        self.icon.setText(icon)
        name = tr(f"tool_{self.tool_id}")
        self.label.setText(f"{tr(key, tool=name)}{(': ' + self.detail) if self.detail else ''}")


class ChatTranscript(QFrame):
    """A scrollable message column shared by AgentPage methods and future controllers."""

    def __init__(self):
        super().__init__()
        self.setObjectName("ChatTranscript")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(SPACING_SM, SPACING_SM, SPACING_SM, SPACING_SM)
        self.layout.setSpacing(SPACING_SM)
        self.layout.addStretch(1)
        self.items: list[QWidget] = []

    def append(self, widget: QWidget) -> QWidget:
        self.layout.insertWidget(self.layout.count() - 1, widget)
        self.items.append(widget)
        return widget

    def clear(self) -> None:
        for item in self.items:
            self.layout.removeWidget(item)
            item.deleteLater()
        self.items.clear()

    def scroll_to_bottom(self, scroll_area) -> None:
        def scroll_if_alive() -> None:
            # A queued chat repaint can outlive a page during application
            # shutdown or project navigation.  It is presentation-only and
            # must never raise into the Qt event loop.
            try:
                bar = scroll_area.verticalScrollBar()
                bar.setValue(bar.maximum())
            except RuntimeError:
                return
        QTimer.singleShot(0, scroll_if_alive)
