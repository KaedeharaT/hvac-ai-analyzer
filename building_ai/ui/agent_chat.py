"""Reusable, presentation-only widgets for the read-only AI Assistant page."""

from __future__ import annotations

from PyQt5.QtCore import Qt, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices
import json

from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget

from building_ai.i18n import LanguageManager, tr
from building_ai.ui.theme import SPACING_SM


class ChatInput(QTextEdit):
    """Multiline editor where Enter submits and Shift+Enter inserts a newline."""

    submitted = pyqtSignal()

    def keyPressEvent(self, event):  # noqa: N802 - Qt API name
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                # QTextEdit's platform handling of Shift+Enter differs under
                # headless Qt backends. Insert explicitly so the interaction
                # remains a predictable multiline input everywhere.
                self.insertPlainText("\n")
                event.accept()
                return
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


class AgentProcessCard(QFrame):
    """Compact user-facing view over the persisted runtime trace."""

    def __init__(self, route=None, plan=None):
        super().__init__(); self.setObjectName("AgentProcessCard")
        self.route = route; self.plan = list(getattr(plan, "steps", []) or [])
        self._completed = False; self._trace: dict = {}; self._elapsed_ms = 0.0
        layout = QVBoxLayout(self); layout.setContentsMargins(SPACING_SM, SPACING_SM, SPACING_SM, SPACING_SM); layout.setSpacing(5)
        top = QHBoxLayout(); self.summary = QLabel(); self.summary.setObjectName("AgentProcessSummary")
        self.toggle = QPushButton(); self.toggle.setObjectName("TextButton"); self.toggle.clicked.connect(self._toggle)
        top.addWidget(self.summary, 1); top.addWidget(self.toggle); layout.addLayout(top)
        self.detail = QFrame(); detail_layout = QVBoxLayout(self.detail); detail_layout.setContentsMargins(4, 2, 4, 2); detail_layout.setSpacing(4)
        self.steps = QLabel(); self.steps.setWordWrap(True); self.steps.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.technical = QTextEdit(); self.technical.setReadOnly(True); self.technical.setVisible(False); self.technical.setMaximumHeight(145)
        self.technical_toggle = QPushButton(); self.technical_toggle.setObjectName("TextButton"); self.technical_toggle.clicked.connect(self._toggle_technical)
        detail_layout.addWidget(self.steps); detail_layout.addWidget(self.technical_toggle); detail_layout.addWidget(self.technical)
        layout.addWidget(self.detail); self.detail.setVisible(False)
        LanguageManager.instance().language_changed.connect(self.retranslate_ui)
        self.retranslate_ui()

    @staticmethod
    def _step_label(tool: str) -> str:
        return tr(f"agent_step_{tool}")

    def retranslate_ui(self) -> None:
        self.toggle.setText(tr("agent_view_process") if not self.detail.isVisible() else tr("agent_hide_process"))
        self.technical_toggle.setText(tr("agent_view_technical") if not self.technical.isVisible() else tr("agent_hide_technical"))
        if self._completed:
            self._render_completed()
        elif self.route:
            self.summary.setText(tr("agent_analyzing_question"))
            labels = [self._step_label(getattr(step, "tool", "")) for step in self.plan]
            self.steps.setText("\n".join(f"● {label}" for label in labels) or tr("agent_understanding"))

    def _toggle(self) -> None:
        self.detail.setVisible(not self.detail.isVisible()); self.retranslate_ui()

    def _toggle_technical(self) -> None:
        self.technical.setVisible(not self.technical.isVisible()); self.retranslate_ui()

    def complete(self, trace: dict, elapsed_ms: float) -> None:
        self._completed = True
        self._trace = trace; self._elapsed_ms = elapsed_ms
        self._render_completed()
        technical = {
            "intent": trace.get("intent"), "plan": trace.get("plan"), "tool_calls": trace.get("tool_calls", []),
            "evidence_checks": trace.get("evidence_checks"), "reflections": trace.get("reflections"),
            "memory_used": trace.get("memory_used"), "knowledge_sources": trace.get("knowledge_sources"),
            "llm_calls": trace.get("llm_calls"), "multi_agent": trace.get("multi_agent"),
        }
        self.technical.setPlainText(json.dumps(technical, ensure_ascii=False, indent=2, default=str))
        self.retranslate_ui()

    def _render_completed(self) -> None:
        calls = self._trace.get("tool_calls", []); sources = self._trace.get("knowledge_sources", [])
        successful = sum(bool(call.get("success")) for call in calls if call.get("tool") != "search_knowledge")
        self.summary.setText(tr("agent_process_complete", tools=successful, sources=len(sources), seconds=self._elapsed_ms / 1000))
        lines = [f"✓ {self._step_label(call.get('tool', ''))}" if call.get("success") else f"⚠ {tr('agent_partial_data')}" for call in calls]
        multi = self._trace.get("multi_agent", {})
        if multi.get("results"):
            labels = {
                "data_analyst": "agent_multi_checked_data", "hvac_expert": "agent_multi_checked_hvac",
                "knowledge": "agent_multi_checked_knowledge", "drawing": "agent_multi_checked_drawing",
                "reviewer": "agent_multi_reviewed",
            }
            lines = [f"✓ {tr(labels[item['agent_role']])}" for item in multi["results"]
                     if item.get("agent_role") in labels and item.get("status") == "SUCCEEDED"]
        if self._trace.get("reflections"):
            lines.extend((f"⚠ {tr('agent_need_more_evidence')}", f"✓ {tr('agent_evidence_completed')}"))
        if sources:
            lines.append(f"✓ {tr('agent_knowledge_checked')}")
        elif any(call.get("tool") == "search_knowledge" for call in calls):
            lines.append(f"○ {tr('agent_knowledge_not_found')}")
        if not lines: lines.append(f"✓ {tr('agent_completed')}")
        self.steps.setText("\n".join(lines))


class AgentSourcesCard(QFrame):
    """Shows actual RAG sources separately from selected-project evidence."""

    def __init__(self, sources: list[dict], trace: dict | None = None):
        super().__init__(); self.setObjectName("AgentSourcesCard")
        self.sources = sources; self.trace = trace or {}
        layout = QVBoxLayout(self); layout.setContentsMargins(SPACING_SM, SPACING_SM, SPACING_SM, SPACING_SM); layout.setSpacing(4)
        self.title = QLabel(); self.title.setObjectName("CardTitle"); layout.addWidget(self.title)
        self.source_labels: list[QLabel] = []
        for source in sources:
            metadata = source.get("metadata", {})
            holder = QFrame(); holder.setObjectName("KnowledgeCitationRow"); holder_layout = QVBoxLayout(holder); holder_layout.setContentsMargins(5, 4, 5, 4); holder_layout.setSpacing(2)
            label = QLabel(); label.setWordWrap(True); label.setTextInteractionFlags(Qt.TextSelectableByMouse); label.setProperty("source", source); self.source_labels.append(label); holder_layout.addWidget(label)
            url = metadata.get("official_url") or source.get("source", "")
            if isinstance(url, str) and url.startswith(("https://", "http://")):
                open_button = QPushButton(); open_button.setObjectName("TextButton"); open_button.setProperty("source_url", url); open_button.clicked.connect(lambda checked=False, value=url: QDesktopServices.openUrl(QUrl(value))); holder_layout.addWidget(open_button)
                open_button.setProperty("citation_open_button", True)
            layout.addWidget(holder)
        details = QTextEdit(); details.setReadOnly(True); details.setVisible(False); details.setMaximumHeight(120)
        details.setPlainText(json.dumps({
            "retrieved_query": self.trace.get("query"), "trace_id": self.trace.get("trace_id"),
            "knowledge_sources": [{"source_id": item.get("metadata", {}).get("source_id"), "chunk_id": item.get("chunk_id"),
                                   "score": item.get("score"), "section": item.get("section"),
                                   "official_url": item.get("metadata", {}).get("official_url"),
                                   "country": item.get("metadata", {}).get("country"), "language": item.get("metadata", {}).get("language")}
                                  for item in sources],
        }, ensure_ascii=False, indent=2))
        self.button = QPushButton(); self.button.setObjectName("TextButton")
        def toggle():
            details.setVisible(not details.isVisible()); self.retranslate_ui()
        self.details = details; self.button.clicked.connect(toggle); layout.addWidget(self.button); layout.addWidget(details)
        LanguageManager.instance().language_changed.connect(self.retranslate_ui); self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self.title.setText(tr("agent_sources"))
        self.button.setText(tr("agent_hide_source_details") if self.details.isVisible() else tr("agent_view_source_details"))
        for label in self.source_labels:
            source = label.property("source") or {}; metadata = source.get("metadata", {})
            title = source.get("title") or source.get("citation") or "—"
            organization = metadata.get("organization") or ""
            section = source.get("section") or ""
            country = metadata.get("country"); language = metadata.get("language")
            location = " · ".join(value for value in (tr("knowledge_country_" + country) if country else "", tr("knowledge_language_" + language) if language else "") if value)
            label.setText("\n".join(value for value in (organization, title, section, location) if value))
        for button in self.findChildren(QPushButton):
            if button.property("citation_open_button"): button.setText(tr("knowledge_view_source"))


class AgentEvidenceCard(QFrame):
    """Separates selected-project evidence from external knowledge sources."""

    def __init__(self, lines: list[str]):
        super().__init__(); self.setObjectName("AgentEvidenceCard")
        layout = QVBoxLayout(self); layout.setContentsMargins(SPACING_SM, SPACING_SM, SPACING_SM, SPACING_SM); layout.setSpacing(3)
        self.title = QLabel(); self.title.setObjectName("CardTitle"); layout.addWidget(self.title)
        body = QLabel("\n".join(f"• {line}" for line in lines)); body.setWordWrap(True); body.setTextInteractionFlags(Qt.TextSelectableByMouse); layout.addWidget(body)
        LanguageManager.instance().language_changed.connect(self.retranslate_ui); self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self.title.setText(tr("agent_project_evidence"))


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
