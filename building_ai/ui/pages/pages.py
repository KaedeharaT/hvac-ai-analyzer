from __future__ import annotations

import json
import logging
import time

import pandas as pd

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QFileDialog, QFormLayout, QFrame, QGridLayout,
    QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QMessageBox,
    QProgressBar, QPushButton, QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)

from building_ai.i18n import LanguageManager, tr
from building_ai.agent_runtime_factory import create_agent_runtime
from building_ai.memory import MemoryStore
from building_ai.observability import TraceStore
from building_ai.core.equipment_identity import normalize_equipment_id
from building_ai.llm import apply_detected_local_model, discover_local_models
from building_ai.models import SemanticStatus, TAXONOMY
from building_ai.storage import DuplicateImportError
from building_ai.ui.theme import SPACING_XS, SPACING_LG, SPACING_MD, SPACING_SM
from building_ai.ui.analysis_renderer import finding_text, opportunity_impact_text, opportunity_priority_text, opportunity_text, reason_text
from building_ai.ui.agent_chat import AgentEvidenceCard, AgentProcessCard, AgentSourcesCard, ChatInput, ChatMessage, ChatTranscript, ToolCallWidget
from building_ai.ui.pages.energy_analysis_page import EnergyAnalysisPage
from building_ai.ui.pages.energy_analysis_page import TimeSeriesChart as EngineeringTimeSeriesChart
from building_ai.ui.components import SectionHeader, StatusBadge
from building_ai.services.product_experience import AnalysisReportService, finding_evidence_summary, passed_checks


LOGGER = logging.getLogger(__name__)


class AgentWorker(QThread):
    tool_event = pyqtSignal(str, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal()

    def __init__(self, handler, message: str):
        super().__init__(); self.handler = handler; self.message = message

    def run(self):
        try:
            self.completed.emit(self.handler(self.message))
        except Exception:
            self.failed.emit()


class AnalysisWorker(QThread):
    """Runs the real analysis pipeline without touching widgets from the worker."""

    progress_changed = pyqtSignal(str, str, object, int, int)
    analysis_finished = pyqtSignal(dict)
    analysis_failed = pyqtSignal(str, str)

    def __init__(self, context):
        super().__init__()
        self.context = context
        self.last_stage = "load_project_data"

    def run(self):
        started = time.monotonic()

        def report(stage, status, equipment=None, current=0, total=0):
            self.last_stage = stage
            self.progress_changed.emit(stage, status, equipment, current, total)

        try:
            result = self.context.run_diagnosis(report)
            kpis = result.analytics.equipment_kpis if result.analytics else []
            finding_equipment = {item.equipment_id for item in result.findings}
            self.analysis_finished.emit({
                "equipment": len(kpis),
                "successful": sum(item.status == "available" for item in kpis),
                "warnings": sum(item.status != "available" for item in kpis),
                "normal": sum(item.status == "available" and item.equipment_id not in finding_equipment for item in kpis),
                "attention": sum(item.status == "available" and item.equipment_id in finding_equipment for item in kpis),
                "findings": len(result.findings),
                "opportunities": len(self.context.opportunities),
                "elapsed": time.monotonic() - started,
            })
        except Exception as exc:
            LOGGER.exception("Analysis pipeline failed at stage %s", self.last_stage)
            self.analysis_failed.emit(self.last_stage, str(exc))


class AnalysisProgressPanel(QFrame):
    """Compact, i18n-rendered view of structured pipeline events."""

    STAGES = (
        "load_project_data", "semantic_mapping", "equipment_grouping", "resolve_signals",
        "calculate_kpi", "diagnostics", "energy_opportunities", "validation", "finalize",
    )
    WEIGHTS = {
        "load_project_data": 10, "semantic_mapping": 20, "equipment_grouping": 15,
        "resolve_signals": 10, "calculate_kpi": 22, "diagnostics": 10,
        "energy_opportunities": 7, "validation": 3, "finalize": 3,
    }
    MARKERS = {"waiting": "○", "running": "●", "completed": "✓", "warning": "⚠", "failed": "✕"}

    def __init__(self):
        super().__init__()
        self.setObjectName("Card")
        self.statuses = {stage: "waiting" for stage in self.STAGES}
        self.records: list[tuple[float, str, str, str | None, int, int]] = []
        self.device_statuses: dict[str, str] = {}
        self.summary: dict | None = None
        self.failure: tuple[str, str] | None = None
        self.started = 0.0
        box = QVBoxLayout(self); box.setContentsMargins(SPACING_MD, SPACING_MD, SPACING_MD, SPACING_MD); box.setSpacing(SPACING_XS)
        self.headline = QLabel(); self.headline.setObjectName("CardTitle"); self.headline.setWordWrap(True); box.addWidget(self.headline)
        self.summary_label = QLabel(); self.summary_label.setObjectName("Muted"); self.summary_label.setWordWrap(True); box.addWidget(self.summary_label)
        self.bar = QProgressBar(); self.bar.setRange(0, 100); self.bar.setTextVisible(True); self.bar.setFixedHeight(14); box.addWidget(self.bar)
        self.current_label = QLabel(); self.current_label.setObjectName("Muted"); box.addWidget(self.current_label)
        self.details_button = QPushButton(); self.details_button.setObjectName("TextButton"); self.details_button.setCheckable(True); self.details_button.toggled.connect(self._toggle_details)
        box.addWidget(self.details_button, alignment=Qt.AlignLeft)
        # Details are intentionally a bounded text view.  The former nested
        # layouts reflowed under translation and caused the visual overlap.
        self.details = QTextEdit(); self.details.setReadOnly(True); self.details.setFixedHeight(156); self.details.hide(); box.addWidget(self.details)
        self.stage_labels = {}
        for stage in self.STAGES:
            self.stage_labels[stage] = QLabel()  # retained as an i18n-testable view model
        LanguageManager.instance().language_changed.connect(self.retranslate_ui)
        self.retranslate_ui()

    def start(self):
        self.started = time.monotonic(); self.summary = None; self.failure = None
        self.statuses = {stage: "waiting" for stage in self.STAGES}; self.records = []; self.device_statuses = {}
        self.details_button.setChecked(False); self.details.hide(); self.show(); self.update_progress("load_project_data", "running")

    def update_progress(self, stage: str, status: str, equipment: str | None = None, current: int = 0, total: int = 0):
        if stage not in self.statuses:
            return
        self.statuses[stage] = status
        if stage == "calculate_kpi" and equipment:
            self.device_statuses[equipment] = status
        self.records.append((time.monotonic(), stage, status, equipment, current, total))
        self._render(current, total, stage)

    def complete(self, summary: dict):
        self.summary = {"normal": 0, "attention": 0, **summary}; self.failure = None
        self.statuses["finalize"] = "completed"; self._render()

    def fail(self, stage: str, error: str):
        self.failure = (stage, error); self.statuses[stage] = "failed"; self._render()

    def _toggle_details(self, checked: bool):
        self.details.setVisible(checked)
        self.details_button.setText(tr("analysis_hide_details") if checked else tr("analysis_view_details"))

    def _render(self, current: int = 0, total: int = 0, active_stage: str | None = None):
        for stage, label in self.stage_labels.items():
            status = self.statuses[stage]
            label.setText(f"{self.MARKERS[status]} {tr('analysis_stage_' + stage)}")
        complete_weight = sum(self.WEIGHTS[stage] for stage, status in self.statuses.items() if status in {"completed", "warning"})
        if active_stage and self.statuses.get(active_stage) == "running" and total:
            complete_weight += self.WEIGHTS[active_stage] * min(current, total) / total
        self.bar.setValue(min(100, round(complete_weight)))
        if self.failure:
            stage, _ = self.failure
            self.headline.setText(f"✕ {tr('analysis_failed_status')}")
            self.summary_label.setText(f"{tr('analysis_failed_stage')}: {tr('analysis_stage_' + stage)}")
        elif self.summary is not None:
            self.headline.setText(f"✓ {tr('analysis_completed')}")
            self.summary_label.setText(tr("analysis_completed_summary", **self.summary))
        else:
            running = next((stage for stage in self.STAGES if self.statuses[stage] == "running"), "load_project_data")
            self.headline.setText(f"● {tr('analysis_running')}")
            self.summary_label.setText(tr("analysis_stage_" + running))
        current_stage = "finalize" if self.summary is not None else next((stage for stage in self.STAGES if self.statuses[stage] == "running"), "finalize")
        recent_device = next(reversed(self.device_statuses), None) if self.device_statuses else None
        current_text = tr("analysis_current_stage", stage=tr("analysis_completed") if self.summary is not None else tr("analysis_stage_" + current_stage))
        if recent_device:
            current_text += f" · {tr('analysis_recent_device', equipment=recent_device)}"
        self.current_label.setText(current_text)
        lines = []
        for timestamp, stage, status, equipment, index, total_value in self.records:
            elapsed = timestamp - self.started if self.started else 0
            suffix = f" · {equipment}" if equipment and stage == "calculate_kpi" else ""
            if total_value and stage == "calculate_kpi": suffix += f" ({index}/{total_value})"
            lines.append(f"{elapsed:05.1f}s  {self.MARKERS[status]} {tr('analysis_stage_' + stage)}{suffix}")
        if self.device_statuses:
            lines.append("")
            lines.extend(f"{self.MARKERS[status]} {equipment} · {tr('analysis_device_' + status)}" for equipment, status in self.device_statuses.items())
        if self.failure:
            lines.append(f"{tr('analysis_error_reason')}: {self.failure[1]}")
        self.details.setText("\n".join(lines))
        self.details_button.setText(tr("analysis_hide_details") if self.details.isVisible() else tr("analysis_view_details"))

    def retranslate_ui(self):
        self._render()


class Card(QFrame):
    def __init__(self, title_key: str, value: str = "—", note: str = ""):
        super().__init__(); self.setObjectName("Card")
        layout = QVBoxLayout(self); layout.setContentsMargins(SPACING_MD, SPACING_MD, SPACING_MD, SPACING_MD); layout.setSpacing(SPACING_SM)
        self.title_key = title_key; self.title = QLabel(); self.title.setObjectName("CardTitle")
        self.value = QLabel(value); self.value.setObjectName("CardValue")
        self.note = QLabel(note); self.note.setObjectName("Muted"); self.note.setWordWrap(True)
        layout.addWidget(self.title); layout.addWidget(self.value); layout.addWidget(self.note); layout.addStretch(1)
        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self.title.setText(tr(self.title_key))

    def set_value(self, value: str, note: str = "") -> None:
        self.value.setText(value); self.note.setText(note)


class MetricBarChart(QWidget):
    """Small dependency-free comparison chart for the analysis dashboard."""

    def __init__(self, value_suffix: str = "", warning_below: float | None = None):
        super().__init__()
        self.value_suffix = value_suffix
        self.warning_below = warning_below
        self.data: list[tuple[str, float | None]] = []
        self.setMinimumHeight(150)

    def set_data(self, data: list[tuple[str, float | None]]):
        self.data = data
        self.update()

    def paintEvent(self, event):  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor("#64748B")))
        rect = self.rect().adjusted(10, 8, -10, -8)
        if not self.data:
            painter.drawText(rect, Qt.AlignCenter, tr("analysis_chart_no_data")); return
        label_width, value_width = 88, 58
        max_value = max((value or 0 for _, value in self.data), default=1.0) or 1.0
        row_height = max(24, rect.height() // max(1, len(self.data)))
        for index, (name, value) in enumerate(self.data):
            y = rect.top() + index * row_height
            painter.setPen(QPen(QColor("#475569")))
            painter.drawText(rect.left(), y, label_width, row_height, Qt.AlignVCenter | Qt.AlignLeft, name)
            if value is None:
                painter.drawText(rect.right() - value_width, y, value_width, row_height, Qt.AlignVCenter | Qt.AlignRight, "N/A")
                continue
            bar_left = rect.left() + label_width + 8
            bar_right = rect.right() - value_width - 8
            bar_width = max(2, int((bar_right - bar_left) * value / max_value))
            warning = self.warning_below is not None and value < self.warning_below
            painter.fillRect(bar_left, y + 6, bar_width, max(10, row_height - 12), QColor("#D97706" if warning else "#2563EB"))
            painter.setPen(QPen(QColor("#172033")))
            painter.drawText(rect.right() - value_width, y, value_width, row_height, Qt.AlignVCenter | Qt.AlignRight, f"{value:.2f}{self.value_suffix}")


class TimeSeriesChart(QWidget):
    """Compact renderer for real service output; it never manufactures points."""
    def __init__(self, kind: str = "line"):
        super().__init__(); self.kind = kind; self.payload = {}; self.setMinimumHeight(190)

    def set_payload(self, payload):
        self.payload = payload or {}; self.update()

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self); painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(32, 12, -12, -26); painter.setPen(QPen(QColor("#64748B")))
        if not self.payload or not (self.payload.get("series") or self.payload.get("data")):
            painter.drawText(rect, Qt.AlignCenter, tr("energy_no_data")); return
        if self.kind == "bar":
            data = self.payload.get("data", []); maximum = max((x.get("value", 0) for x in data), default=1) or 1
            height = max(18, rect.height() // max(1, len(data)))
            for i, item in enumerate(data):
                y = rect.top() + i * height; width = int((rect.width() - 105) * item["value"] / maximum)
                painter.setPen(QPen(QColor("#475569"))); painter.drawText(rect.left(), y, 90, height, Qt.AlignVCenter, item["name"])
                painter.fillRect(rect.left() + 94, y + 5, width, max(8, height - 10), QColor("#2563EB"))
                painter.drawText(rect.right() - 64, y, 64, height, Qt.AlignVCenter | Qt.AlignRight, f"{item['value']:.1f}")
            return
        if self.kind == "scatter":
            data = self.payload.get("data", []); xs = [x["x"] for x in data]; ys = [x["y"] for x in data]
            if not xs or max(xs) == min(xs) or max(ys) == min(ys): painter.drawText(rect, Qt.AlignCenter, tr("energy_no_data")); return
            painter.drawLine(rect.left(), rect.bottom(), rect.right(), rect.bottom()); painter.drawLine(rect.left(), rect.top(), rect.left(), rect.bottom())
            painter.setPen(QPen(QColor("#2563EB"), 3))
            for x, y in zip(xs, ys): painter.drawPoint(rect.left() + int((x-min(xs))/(max(xs)-min(xs))*rect.width()), rect.bottom() - int((y-min(ys))/(max(ys)-min(ys))*rect.height()))
            return
        if self.kind == "heatmap":
            data = self.payload.get("data", []); values = [x["value"] for x in data]; maximum = max(values) if values else 0
            dates = list(dict.fromkeys(x["date"] for x in data)); times = list(dict.fromkeys(x["time"] for x in data))
            if not maximum or not dates or not times: painter.drawText(rect, Qt.AlignCenter, tr("energy_no_data")); return
            lookup = {(x["date"], x["time"]): x["value"] for x in data}; width = max(1, rect.width() // len(dates)); height = max(1, rect.height() // len(times))
            for ix, day in enumerate(dates):
                for iy, clock in enumerate(times):
                    value = lookup.get((day, clock)); ratio = 0 if value is None else value / maximum
                    painter.fillRect(rect.left()+ix*width, rect.top()+iy*height, width, height, QColor(int(240-130*ratio), int(248-115*ratio), int(255-25*ratio)))
            return
        series = self.payload.get("series", []); palette = ("#2563EB", "#D97706", "#16A34A", "#9333EA")
        all_values = [point["value"] for item in series for point in item.get("data", []) if point.get("value") is not None]
        if not all_values: painter.drawText(rect, Qt.AlignCenter, tr("energy_no_data")); return
        low, high = min(all_values), max(all_values); span = high - low or 1
        for si, item in enumerate(series):
            points = item.get("data", []); painter.setPen(QPen(QColor(palette[si % len(palette)]), 2)); previous = None
            for ix, point in enumerate(points):
                if point.get("value") is None: previous = None; continue
                x = rect.left() + int(ix * rect.width() / max(1, len(points)-1)); y = rect.bottom() - int((point["value"]-low)/span*rect.height())
                if previous: painter.drawLine(previous[0], previous[1], x, y)
                previous = (x, y)
            painter.drawText(rect.left()+si*120, rect.bottom()+4, 115, 16, Qt.AlignLeft, item.get("name", ""))
        painter.setPen(QPen(QColor("#64748B"))); painter.drawText(2, rect.top(), 28, 18, Qt.AlignRight, f"{high:.1f}"); painter.drawText(2, rect.bottom()-10, 28, 18, Qt.AlignRight, f"{low:.1f}")


class BasePage(QWidget):
    def __init__(self, context, title_key: str):
        super().__init__()
        self.context = context; self.title_key = title_key
        self.layout = QVBoxLayout(self); self.layout.setContentsMargins(SPACING_LG, SPACING_LG, SPACING_LG, SPACING_LG); self.layout.setSpacing(SPACING_MD)
        self.i18n = LanguageManager.instance(); self.i18n.language_changed.connect(self.retranslate_ui)
        # The persistent application header already carries the page title.
        # Retain this label for compatibility and translations, but avoid a
        # second visible title in the page body.
        self.heading = QLabel(); self.heading.setObjectName("PageTitle"); self.heading.hide(); self.layout.addWidget(self.heading)
        self.heading.setText(tr(self.title_key))

    def retranslate_ui(self) -> None:
        self.heading.setText(tr(self.title_key))

    @staticmethod
    def setup_table(table: QTableWidget) -> None:
        table.setAlternatingRowColors(True); table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers); table.verticalHeader().setDefaultSectionSize(34)
        table.horizontalHeader().setStretchLastSection(True)


class DashboardPage(BasePage):
    navigation_requested = pyqtSignal(str, object)

    def __init__(self, context):
        super().__init__(context, "dashboard")
        self.subtitle = QLabel(); self.subtitle.setObjectName("Muted"); self.layout.addWidget(self.subtitle)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True); self.scroll.setFrameShape(QFrame.NoFrame)
        self.body = QWidget(); self.body_layout = QVBoxLayout(self.body); self.body_layout.setContentsMargins(0, 0, 0, 0); self.body_layout.setSpacing(SPACING_MD)
        self.scroll.setWidget(self.body); self.layout.addWidget(self.scroll, 1)
        grid = QGridLayout(); grid.setHorizontalSpacing(SPACING_MD); grid.setVerticalSpacing(SPACING_MD)
        self.cards = {
            "readiness": Card("analysis_ready"), "equipment": Card("analysis_equipment_total"),
            "energy": Card("energy_total"), "attention": Card("analysis_attention_equipment"),
        }
        for index, card in enumerate(self.cards.values()):
            grid.addWidget(card, 0, index)
        self.body_layout.addLayout(grid)
        readiness_card = QFrame(); readiness_card.setObjectName("Card"); readiness_box = QVBoxLayout(readiness_card)
        self.readiness_header = SectionHeader(tr("dashboard_readiness"), ""); readiness_box.addWidget(self.readiness_header)
        self.readiness_steps = QHBoxLayout(); readiness_box.addLayout(self.readiness_steps); self.body_layout.addWidget(readiness_card)
        trend_card = QFrame(); trend_card.setObjectName("ChartCard"); trend_box = QVBoxLayout(trend_card); trend_box.setContentsMargins(SPACING_MD, SPACING_MD, SPACING_MD, SPACING_MD)
        trend_box.addWidget(SectionHeader(tr("energy_chart_energy"), tr("page_subtitle_energy_analysis")))
        self.energy_chart = EngineeringTimeSeriesChart("line"); self.energy_chart.setMinimumHeight(285); trend_box.addWidget(self.energy_chart)
        lower = QHBoxLayout(); lower.setSpacing(SPACING_MD)
        attention = QFrame(); attention.setObjectName("Card"); attention_box = QVBoxLayout(attention); attention_box.addWidget(SectionHeader(tr("dashboard_attention"), ""))
        self.attention_table = QTableWidget(0, 3); self.setup_table(self.attention_table); self.attention_table.setHorizontalHeaderLabels([tr("equipment"), tr("diagnostics_finding_column"), tr("status")]); self.attention_table.setMaximumHeight(190); self.attention_table.cellDoubleClicked.connect(self._open_finding)
        self.attention_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.attention_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.attention_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        attention_box.addWidget(self.attention_table)
        next_card = QFrame(); next_card.setObjectName("Card"); next_box = QVBoxLayout(next_card); next_box.addWidget(SectionHeader(tr("dashboard_next_check"), "")); self.next_check = QLabel(); self.next_check.setWordWrap(True); next_box.addWidget(self.next_check); next_box.addStretch(1)
        self.ask_button = QPushButton(tr("ask_ai")); self.ask_button.setObjectName("PrimaryButton"); self.ask_button.clicked.connect(lambda: self.navigation_requested.emit("agent", {"prompt": tr("agent_question_findings")})); next_box.addWidget(self.ask_button)
        lower.addWidget(attention, 2); lower.addWidget(next_card, 1); self.body_layout.addLayout(lower)
        self.body_layout.addWidget(trend_card)
        self.refresh()

    def retranslate_ui(self) -> None:
        super().retranslate_ui()
        if hasattr(self, "subtitle"):
            self.subtitle.setText(tr("page_subtitle_dashboard"))
        for card in getattr(self, "cards", {}).values(): card.retranslate_ui()
        self.refresh()

    def refresh(self):
        project = self.context.current_project; semantics = self.context.semantic_result
        energy = getattr(self.context, "energy_analysis_result", None)
        counts = {status.value: 0 for status in SemanticStatus}
        if semantics:
            for item in semantics.semantic_results: counts[item.status.value] += 1
        imported = self.context.dataframe is not None
        mapped = bool(semantics and semantics.semantic_results)
        equipment_count = len(self.context.equipment)
        analysis_ready = self.context.diagnosis_result is not None
        self.cards["readiness"].set_value(tr("analysis_ready") if analysis_ready else tr("analysis_not_ready"))
        self.cards["equipment"].set_value(str(equipment_count))
        self.cards["energy"].set_value(f"{energy.summary.get('total_energy_kwh', 0):,.1f} kWh" if energy and energy.summary.get("total_energy_kwh") is not None else "—")
        diagnosis = self.context.diagnosis_result
        finding_count = len(diagnosis.findings) if diagnosis else 0
        self.cards["attention"].set_value(str(len({item.equipment_id for item in diagnosis.findings}) if diagnosis else 0))
        metadata = [project.name if project else tr("no_project"), f"{self.context.import_metadata.get('start', '—')} — {self.context.import_metadata.get('end', '—')}"]
        if semantics:
            metadata.append(f"{len(semantics.semantic_results)} {tr('points')}")
        self.subtitle.setText(" · ".join(metadata))
        chart = dict(energy.charts.get("energy_trend") or {}) if energy else {}
        if chart:
            chart["x_label"] = tr("energy_axis_time"); chart["y_label"] = tr("energy_axis_interval_energy")
        self.energy_chart.set_payload(chart)
        while self.readiness_steps.count():
            child = self.readiness_steps.takeAt(0)
            if child.widget(): child.widget().deleteLater()
        statuses = (
            ("readiness_import", imported), ("readiness_semantics", mapped),
            ("readiness_review", not bool(counts.get("REVIEW") or counts.get("ABSTAIN"))),
            ("readiness_equipment", equipment_count > 0), ("readiness_analysis", analysis_ready),
        )
        for key, ready in statuses:
            frame = QFrame(); frame.setObjectName("ReadinessStep"); row = QHBoxLayout(frame); row.setContentsMargins(SPACING_SM, SPACING_SM, SPACING_SM, SPACING_SM)
            status = QLabel(tr("readiness_complete") if ready else tr("readiness_pending")); status.setObjectName("Muted")
            row.addWidget(status); row.addWidget(QLabel(tr(key))); self.readiness_steps.addWidget(frame, 1)
        findings = list(diagnosis.findings if diagnosis else [])
        names = {item.equipment_id: item.equipment_name for item in diagnosis.analytics.equipment_kpis} if diagnosis and diagnosis.analytics else {}
        self.attention_table.setRowCount(len(findings))
        for row, item in enumerate(findings):
            values = [names.get(item.equipment_id, item.equipment_id or "—"), finding_text(item)[0], tr("finding_severity_" + item.severity)]
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value); cell.setData(Qt.UserRole, item.finding_id); self.attention_table.setItem(row, column, cell)
        user_items = getattr(self.context, "user_interpretations", [])
        if user_items:
            item = user_items[0]
            self.next_check.setText(f"{item.problem}\n\n{tr('recommendation_first_action')}{item.actions[0]}\n\n{tr('diagnostics_verification')}: {item.expected_effect}")
        else:
            self.next_check.setText(tr("dashboard_no_active") + "\n\n" + tr("dashboard_not_health_claim"))

    def _open_finding(self, row: int, column: int) -> None:
        item = self.attention_table.item(row, 0)
        if item:
            self.navigation_requested.emit("analysis", {"finding_id": item.data(Qt.UserRole), "equipment_id": item.text()})


class ProjectsPage(BasePage):
    project_changed = pyqtSignal()
    def __init__(self, context):
        super().__init__(context, "projects")
        buttons = QHBoxLayout(); self.create_button = QPushButton(); self.create_button.setObjectName("PrimaryButton"); self.open_button = QPushButton(); self.rename_button = QPushButton(); self.delete_button = QPushButton(); self.delete_button.setObjectName("DangerButton")
        for button, fn in ((self.create_button, self.create), (self.open_button, self.open), (self.rename_button, self.rename), (self.delete_button, self.delete)):
            button.clicked.connect(fn); buttons.addWidget(button)
        buttons.addStretch(1); self.layout.addLayout(buttons)
        self.list = QTableWidget(0, 5); self.setup_table(self.list); self.layout.addWidget(self.list, 1); self.refresh(); self.retranslate_ui()

    def retranslate_ui(self):
        super().retranslate_ui()
        if hasattr(self, "create_button"):
            self.create_button.setText(tr("create")); self.open_button.setText(tr("open")); self.rename_button.setText(tr("rename")); self.delete_button.setText(tr("delete"))
            self.list.setHorizontalHeaderLabels([tr("project_name_column"), tr("time_range"), tr("project_equipment_count"), tr("project_data_status"), tr("project_last_analysis")])

    def refresh(self):
        if not hasattr(self, "list"): return
        projects = self.context.projects.list(); self.list.setRowCount(len(projects))
        for row, project in enumerate(projects):
            imported = bool(project.source_files)
            values = [
                project.name, project.time_range or "—", str(len({item.effective_equipment_id for item in self.context.projects.load_semantics(project.project_id) if item.effective_equipment_id})),
                tr("data_ready") if imported else tr("data_missing"),
                project.analysis_summary.get("status", "—") if project.analysis_summary else "—",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 0: item.setData(Qt.UserRole, project.project_id)
                if column in (2, 3, 4): item.setTextAlignment(Qt.AlignCenter)
                self.list.setItem(row, column, item)

    def selected_id(self):
        item = self.list.item(self.list.currentRow(), 0) if self.list.currentRow() >= 0 else None
        return item.data(Qt.UserRole) if item else None

    def create(self):
        name, ok = QInputDialog.getText(self, tr("create_project"), tr("project_name"))
        if ok and name.strip(): self.context.current_project = self.context.projects.create(name.strip()); self.refresh(); self.project_changed.emit()

    def open(self):
        project_id = self.selected_id()
        if project_id: self.context.open_project(project_id); self.project_changed.emit()

    def rename(self):
        project_id = self.selected_id()
        if project_id:
            name, ok = QInputDialog.getText(self, tr("rename_project"), tr("new_name"))
            if ok and name.strip(): self.context.current_project = self.context.projects.rename(project_id, name.strip()); self.refresh(); self.project_changed.emit()

    def delete(self):
        project_id = self.selected_id()
        if project_id and QMessageBox.question(self, tr("delete_project"), tr("delete_project_confirm")) == QMessageBox.Yes:
            self.context.project_data.clear(project_id)
            self.context.projects.delete(project_id)
            if self.context.current_project and self.context.current_project.project_id == project_id: self.context.reset_project()
            self.refresh(); self.project_changed.emit()


class DataPage(BasePage):
    data_changed = pyqtSignal()
    def __init__(self, context):
        super().__init__(context, "import_data")
        buttons = QHBoxLayout(); self.import_button = QPushButton(); self.import_button.setObjectName("PrimaryButton"); self.replace_button = QPushButton(); self.clear_button = QPushButton(); self.clear_button.setObjectName("DangerButton")
        self.import_button.clicked.connect(lambda: self.import_file("add")); self.replace_button.clicked.connect(lambda: self.import_file("replace")); self.clear_button.clicked.connect(self.clear_data)
        for button in (self.import_button, self.replace_button, self.clear_button): buttons.addWidget(button)
        buttons.addStretch(1); self.layout.addLayout(buttons)
        self.readiness = QLabel(); self.readiness.setObjectName("Muted"); self.readiness.setWordWrap(True); self.layout.addWidget(self.readiness)
        self.metadata = QTextEdit(); self.metadata.setReadOnly(True); self.metadata.setObjectName("Card"); self.metadata.setMaximumHeight(125); self.layout.addWidget(self.metadata)
        self.preview = QTableWidget(); self.setup_table(self.preview); self.layout.addWidget(self.preview, 1); self.retranslate_ui()

    def retranslate_ui(self):
        super().retranslate_ui()
        if hasattr(self, "import_button"):
            self.import_button.setText(tr("add_data")); self.replace_button.setText(tr("replace_data")); self.clear_button.setText(tr("clear_project_data")); self.refresh()

    def import_file(self, mode: str = "add"):
        if not self.context.current_project: QMessageBox.warning(self, tr("no_project"), tr("no_project_message")); return
        if mode == "replace" and QMessageBox.question(self, tr("replace_data"), tr("replace_data_confirm"), QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes: return
        path, _ = QFileDialog.getOpenFileName(self, tr("import"), "", "Data (*.csv *.xlsx *.xls)")
        if not path: return
        sheets = self.context.importer.list_sheets(path); sheet = None
        if len(sheets) > 1:
            choices = [*sheets, "__all__"]
            sheet, ok = QInputDialog.getItem(self, tr("worksheet"), tr("select_sheet"), choices, 0, False)
            if not ok: return
        try:
            result = self.context.import_data(path, sheet, mode=mode)
        except DuplicateImportError:
            duplicate = QMessageBox.question(self, tr("duplicate_import"), tr("duplicate_import_message"), QMessageBox.Yes | QMessageBox.No)
            if duplicate != QMessageBox.Yes: return
            result = self.context.import_data(path, sheet, mode=mode, allow_duplicate=True)
        self.refresh()
        frame = result.dataframe.head(50); self.preview.setRowCount(len(frame)); self.preview.setColumnCount(len(frame.columns)); self.preview.setHorizontalHeaderLabels([str(c) for c in frame.columns])
        for row in range(len(frame)):
            for col in range(len(frame.columns)):
                item = QTableWidgetItem(str(frame.iat[row, col])); item.setTextAlignment(Qt.AlignRight if hasattr(frame.iat[row, col], "real") else Qt.AlignLeft | Qt.AlignVCenter); self.preview.setItem(row, col, item)
        self.data_changed.emit()

    def clear_data(self):
        if not self.context.current_project: return
        if QMessageBox.question(self, tr("clear_project_data"), tr("clear_data_confirm"), QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes: return
        self.context.clear_project_data(); self.refresh(); self.data_changed.emit()

    def refresh(self):
        if not hasattr(self, "metadata"): return
        project = self.context.current_project
        if not project:
            self.metadata.setText(tr("no_project_message")); self.readiness.setText("○ " + tr("readiness_import")); return
        if self.context.dataframe is None:
            self.metadata.setText(tr(self.context.data_notice or "no_project_data")); self.readiness.setText("○ " + tr("readiness_import")); self.preview.setRowCount(0); self.preview.setColumnCount(0); return
        info = self.context.import_metadata
        latest = (info.get("imports") or [{}])[-1]
        summary = [
            tr("import_summary"),
            f"{tr('import_files')}: {len(info.get('imports', []))}",
            f"{tr('points')}: {len(self.context.dataframe.columns)} · {tr('import_rows')}: {len(self.context.dataframe):,}",
            f"{tr('time_range')}: {project.time_range or '—'}",
            f"{tr('import_last_updated')}: {latest.get('imported_at') or '—'} · {tr('import_revision')}: {project.data_revision}",
        ]
        self.metadata.setPlainText("\n".join(summary))
        semantics = self.context.semantic_result.semantic_results if self.context.semantic_result else []
        mapped = sum(item.status.value == "ACCEPT" for item in semantics); review = sum(item.status.value == "REVIEW" for item in semantics); abstained = sum(item.status.value == "ABSTAIN" for item in semantics)
        self.readiness.setText(f"✓ {tr('readiness_import')}  →  {'✓' if semantics else '○'} {tr('readiness_semantics')} ({mapped} {tr('mapped')} · {review} {tr('review')} · {abstained} {tr('abstained')})  →  {'✓' if self.context.equipment else '○'} {tr('readiness_equipment')}")
        frame = self.context.dataframe.head(50); self.preview.setRowCount(len(frame)); self.preview.setColumnCount(len(frame.columns)); self.preview.setHorizontalHeaderLabels([str(c) for c in frame.columns])
        for row in range(len(frame)):
            for col in range(len(frame.columns)):
                item = QTableWidgetItem(str(frame.iat[row, col])); item.setTextAlignment(Qt.AlignRight if hasattr(frame.iat[row, col], "real") else Qt.AlignLeft | Qt.AlignVCenter); self.preview.setItem(row, col, item)


class SemanticsPage(BasePage):
    def __init__(self, context):
        super().__init__(context, "semantic_mapping")
        self.run_button = QPushButton(); self.run_button.setObjectName("PrimaryButton"); self.run_button.clicked.connect(self.run_analysis); self.layout.addWidget(self.run_button, alignment=Qt.AlignLeft)
        self.summary = QLabel(); self.summary.setObjectName("Muted"); self.layout.addWidget(self.summary)
        self.table = QTableWidget(0, 10); self.setup_table(self.table); self.table.itemSelectionChanged.connect(self.show_details); self.layout.addWidget(self.table, 1)
        review = QHBoxLayout(); self.label = QComboBox(); self.label.addItems(TAXONOMY); self.equipment_id = QLineEdit(); self.note = QLineEdit(); self.accept = QPushButton(); self.accept.setObjectName("PrimaryButton"); self.unknown = QPushButton()
        self.accept.clicked.connect(self.save_review); self.unknown.clicked.connect(self.mark_unknown)
        for widget in (self.label, self.equipment_id, self.note, self.accept, self.unknown): review.addWidget(widget)
        self.layout.addLayout(review); self.details = QTextEdit(); self.details.setReadOnly(True); self.details.setMaximumHeight(130); self.layout.addWidget(self.details); self.retranslate_ui()

    def retranslate_ui(self):
        super().retranslate_ui()
        if hasattr(self, "table"):
            self.run_button.setText(tr("run_semantic")); self.table.setHorizontalHeaderLabels([tr("raw_point"), tr("canonical_label"), tr("device_type"), tr("equipment_id"), tr("physical_quantity"), tr("unit"), tr("confidence"), tr("validation"), tr("status"), tr("source")]); self.equipment_id.setPlaceholderText(tr("equipment_id")); self.note.setPlaceholderText(tr("review_note")); self.accept.setText(tr("confirm_mapping")); self.unknown.setText(tr("mark_unknown"))

    def run_analysis(self):
        if self.context.dataframe is None: QMessageBox.warning(self, tr("no_data"), tr("import_first")); return
        self.context.run_semantics(); self.refresh()

    def refresh(self):
        if not hasattr(self, "table"): return
        items = self.context.semantic_result.semantic_results if self.context.semantic_result else []; self.table.setRowCount(len(items))
        counts = {"ACCEPT": 0, "REVIEW": 0, "ABSTAIN": 0}
        for item in items: counts[item.status.value] = counts.get(item.status.value, 0) + 1
        self.summary.setText(f"{len(items)} {tr('points')} · {counts['ACCEPT']} {tr('mapped')} · {counts['REVIEW']} {tr('review')} · {counts['ABSTAIN']} {tr('abstained')} · {len(self.context.equipment)} {tr('equipment')}")
        for row, item in enumerate(items):
            source = item.confirmation_source or item.debug_metadata.get("llm_prior", {}).get("source", "engineering_offline")
            values = [item.raw_name, item.effective_label, item.equipment_type or "", item.effective_equipment_id or "", item.physical_quantity or "", item.unit or "", "" if item.confidence is None else f"{item.confidence:.2f}", "valid" if item.physical_validity else "check", item.review_status, source]
            for col, value in enumerate(values): self.table.setItem(row, col, QTableWidgetItem(str(value)))

    def current_result(self):
        row = self.table.currentRow(); items = self.context.semantic_result.semantic_results if self.context.semantic_result else []; return items[row] if 0 <= row < len(items) else None

    def show_details(self):
        item = self.current_result()
        if item:
            self.label.setCurrentText(item.effective_label); self.equipment_id.setText(item.effective_equipment_id or ""); self.details.setText(json.dumps({"reason": item.reason, "equipment_id": item.effective_equipment_id, "relation_confidence": item.relation_confidence, "physics_warnings": item.physics_warnings, "gate_status": item.gate_status, "suspicious": item.suspicious, "per_label_scores": item.per_label_scores}, indent=2, ensure_ascii=False, default=str))

    def save_review(self):
        item = self.current_result()
        if item: self.context.save_review(item.point_id, self.label.currentText(), self.note.text(), self.equipment_id.text().strip() or None); self.refresh()

    def mark_unknown(self): self.label.setCurrentText("other"); self.save_review()


class EquipmentPage(BasePage):
    navigation_requested = pyqtSignal(str, object)

    def __init__(self, context):
        super().__init__(context, "equipment")
        self.splitter = QSplitter(); self.layout.addWidget(self.splitter, 1)
        self.table = QTableWidget(0, 4); self.setup_table(self.table); self.table.itemSelectionChanged.connect(self._render_detail); self.splitter.addWidget(self.table)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        detail = QFrame(); detail.setObjectName("FindingDetail"); box = QVBoxLayout(detail)
        self.detail_title = QLabel(tr("equipment_select")); self.detail_title.setObjectName("SectionTitle"); box.addWidget(self.detail_title)
        self.detail_meta = QLabel(); self.detail_meta.setObjectName("Muted"); self.detail_meta.setWordWrap(True); box.addWidget(self.detail_meta)
        buttons = QHBoxLayout(); self.energy_button = QPushButton(); self.ai_button = QPushButton(); self.ai_button.setObjectName("PrimaryButton"); self.drawing_button = QPushButton()
        self.energy_button.clicked.connect(lambda: self._navigate("energy_analysis")); self.ai_button.clicked.connect(lambda: self._navigate("agent")); self.drawing_button.clicked.connect(lambda: self._navigate("drawing_intelligence"))
        for button in (self.energy_button, self.ai_button, self.drawing_button): buttons.addWidget(button)
        buttons.addStretch(1); box.addLayout(buttons)
        self.tabs = QTabWidget(); self.detail_views = {}
        for key in ("equipment_overview", "equipment_performance", "equipment_diagnostics", "equipment_signals", "equipment_drawing", "equipment_ai_insight"):
            view = QTextEdit(); view.setReadOnly(True); self.detail_views[key] = view; self.tabs.addTab(view, "")
        box.addWidget(self.tabs, 1); self.splitter.addWidget(detail); self.splitter.setSizes([460, 650]); self.retranslate_ui(); self.refresh()

    def retranslate_ui(self):
        super().retranslate_ui()
        if hasattr(self, "table"):
            self.table.setHorizontalHeaderLabels([tr("equipment_name"), tr("device_type"), tr("equipment_analysis_coverage_short"), tr("equipment_finding_count_short")])
            for index, key in enumerate(self.detail_views): self.tabs.setTabText(index, tr(key))
            self.energy_button.setText(tr("equipment_open_energy")); self.ai_button.setText(tr("equipment_ask_ai")); self.drawing_button.setText(tr("equipment_view_drawing"))

    def refresh(self):
        if not hasattr(self, "table"): return
        selected = self.context.selected_equipment_id
        diagnosis = self.context.diagnosis_result
        kpis = diagnosis.analytics.equipment_kpis if diagnosis and diagnosis.analytics else []
        findings = diagnosis.findings if diagnosis else []
        self.table.setRowCount(len(self.context.equipment))
        for row, equipment in enumerate(self.context.equipment):
            kpi = next((item for item in kpis if item.equipment_id == equipment.equipment_id or item.equipment_name == equipment.name), None)
            count = sum(item.equipment_id == equipment.equipment_id for item in findings)
            values = [equipment.name, equipment.equipment_type.value, tr("analysis_available") if kpi and kpi.status == "available" else tr("analysis_skipped"), str(count)]
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value); cell.setData(Qt.UserRole, equipment.name); self.table.setItem(row, column, cell)
            if equipment.name == selected: self.table.selectRow(row)
        if self.table.rowCount() and self.table.currentRow() < 0: self.table.selectRow(0)
        if self.table.rowCount():
            # Rebuilding a QTableWidget can preserve a selected row without
            # emitting itemSelectionChanged. Render explicitly so cross-page
            # equipment context never leaves a stale detail panel.
            self._render_detail()
        if not self.table.rowCount():
            self.detail_title.setText(tr("equipment_select"));
            for view in self.detail_views.values(): view.setPlainText(tr("run_semantics_first"))

    def apply_global_context(self):
        self.refresh()

    def _selected_equipment(self):
        row = self.table.currentRow(); name = self.table.item(row, 0).data(Qt.UserRole) if row >= 0 and self.table.item(row, 0) else None
        return next((item for item in self.context.equipment if item.name == name), None)

    def _render_detail(self):
        equipment = self._selected_equipment()
        if not equipment: return
        self.context.selected_equipment_id = equipment.name
        diagnosis = self.context.diagnosis_result; kpis = diagnosis.analytics.equipment_kpis if diagnosis and diagnosis.analytics else []
        kpi = next((item for item in kpis if item.equipment_id == equipment.equipment_id or item.equipment_name == equipment.name), None)
        findings = [item for item in (diagnosis.findings if diagnosis else []) if item.equipment_id == equipment.equipment_id]
        checks = [item for item in passed_checks(diagnosis) if item.equipment_id == equipment.equipment_id]
        locations = self.context.drawings.equipment_location(self.context.current_project.project_id, equipment.equipment_id) if self.context.current_project else []
        self.detail_title.setText(equipment.name)
        self.detail_meta.setText(f"{equipment.equipment_type.value} · {tr('equipment_analysis_coverage')}: {tr('analysis_available') if kpi and kpi.status == 'available' else tr('analysis_skipped')} · {tr('equipment_finding_count')}: {len(findings)}")
        self.detail_views["equipment_overview"].setPlainText((tr("equipment_no_finding") + "\n" + tr("equipment_no_finding_note")) if not findings else "\n".join(f"• {finding_text(item)[0]}" for item in findings))
        metrics = kpi.metric_summary if kpi else {}
        metric_lines = []
        for label, key, unit in (("Power", "power_kw", "kW"), ("COP", "cop", ""), ("ΔT", "delta_t_c", "°C"), ("Thermal load", "thermal_load_kw", "kW")):
            value = metrics.get(key, {}).get("mean"); metric_lines.append(f"{label}: {'N/A' if value is None else f'{value:,.2f} {unit}'.strip()}")
        self.detail_views["equipment_performance"].setPlainText("\n".join(metric_lines) + (f"\n\n{tr('analysis_reason')}: {reason_text(kpi.reason)}" if kpi and kpi.status != "available" else ""))
        finding_lines = [f"{finding_text(item)[0]}\n{finding_evidence_summary(item)}" for item in findings]
        if checks: finding_lines.extend([f"✓ {item.title}\n{tr('diagnostics_passed_note')}" for item in checks])
        self.detail_views["equipment_diagnostics"].setPlainText("\n\n".join(finding_lines) or tr("analysis_no_findings"))
        binding = next((item for item in (self.context.equipment_organization.heat_sources if self.context.equipment_organization else []) if item.equipment_id == equipment.equipment_id), None)
        self.detail_views["equipment_signals"].setPlainText("\n".join(f"{role}: {point.raw_name} [{point.unit or '—'}]" for role, point in (binding.points_by_role.items() if binding else [])) or tr("analysis_missing_signals", fields="—"))
        self.detail_views["equipment_drawing"].setPlainText("\n".join(tr("equipment_drawing_record", file=item['file_name'], page=item['page_number'], object_class=item['reviewed_class'] or item['class_name']) for item in locations) or tr("agent_drawing_no_reliable_mapping"))
        self.detail_views["equipment_ai_insight"].setPlainText(tr("equipment_ask_ai") + "\n\n" + tr("agent_question_findings") + "\n" + tr("agent_question_cop"))

    def _navigate(self, key: str):
        equipment = self._selected_equipment()
        if equipment:
            prompt = tr("equipment_investigation_prompt", equipment=equipment.name) if key == "agent" else None
            self.navigation_requested.emit(key, {"equipment_id": equipment.name, "prompt": prompt})


class LegacyEnergyAnalysisPage(BasePage):
    """Data-first view: only analyses supported by the current project appear."""
    CHARTS = (
        ("energy_trend", "energy_chart_energy", "line"), ("power_trend", "energy_chart_power", "line"),
        ("temperature_trend", "energy_chart_temperature", "line"), ("delta_t_trend", "energy_chart_delta_t", "line"),
        ("cop_trend", "energy_chart_cop", "line"), ("daily_load_profile", "energy_chart_profile", "line"),
        ("load_heatmap", "energy_chart_heatmap", "heatmap"), ("weather_correlation", "energy_chart_weather", "scatter"),
        ("equipment_ranking", "energy_chart_ranking", "bar"),
    )
    def __init__(self, context):
        super().__init__(context, "energy_analysis")
        controls = QHBoxLayout(); self.range_label = QLabel(); self.range = QComboBox(); self.range.addItems(["all", "24h", "7d", "30d"])
        self.equipment_label = QLabel(); self.equipment = QComboBox(); self.range.currentIndexChanged.connect(self.refresh); self.equipment.currentIndexChanged.connect(self.refresh)
        for widget in (self.range_label, self.range, self.equipment_label, self.equipment): controls.addWidget(widget)
        controls.addStretch(1); self.layout.addLayout(controls)
        self.metadata = QLabel(); self.metadata.setObjectName("Muted"); self.metadata.setWordWrap(True); self.layout.addWidget(self.metadata)
        self.cards_grid = QGridLayout(); self.cards = {key: Card(title) for key, title in (("energy", "energy_total"), ("peak", "energy_peak"), ("cop", "analysis_cop"), ("dt", "analysis_delta_t"))}
        for i, card in enumerate(self.cards.values()): self.cards_grid.addWidget(card, 0, i)
        self.layout.addLayout(self.cards_grid)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True); self.scroll.setFrameShape(QFrame.NoFrame)
        self.content = QWidget(); self.chart_layout = QVBoxLayout(self.content); self.chart_layout.setSpacing(SPACING_MD); self.chart_layout.addStretch(1); self.scroll.setWidget(self.content); self.layout.addWidget(self.scroll, 1)
        self.retranslate_ui()

    def retranslate_ui(self):
        super().retranslate_ui()
        if hasattr(self, "range"):
            current = self.range.currentData() or self.range.currentText(); self.range.blockSignals(True); self.range.clear()
            for key in ("all", "24h", "7d", "30d"): self.range.addItem(tr("energy_range_" + key), key)
            self.range.setCurrentIndex(max(0, self.range.findData(current))); self.range.blockSignals(False)
            self.range_label.setText(tr("energy_time_range")); self.equipment_label.setText(tr("analysis_equipment_filter"))
            for card in self.cards.values(): card.retranslate_ui()
            self.refresh()

    def _result(self):
        base = self.context.energy_analysis_result
        if not base or self.context.dataframe is None or not self.context.semantic_result: return None
        frame = self.context.dataframe; time_name = self.context.import_metadata.get("time_column")
        option = self.range.currentData(); selected = self.equipment.currentData()
        if time_name in frame.columns and option != "all":
            timestamps = pd.to_datetime(frame[time_name], errors="coerce"); hours = {"24h": 24, "7d": 168, "30d": 720}[option]
            frame = frame.loc[timestamps >= timestamps.max() - pd.Timedelta(hours=hours)].copy()
        analytics = self.context.diagnosis_result.analytics if self.context.diagnosis_result else None
        return self.context.energy_analysis.analyze(frame, self.context.semantic_result, self.context.current_project.project_id, self.context.import_metadata, self.context.equipment_organization, analytics, selected)

    def refresh(self):
        if not hasattr(self, "chart_layout"): return
        base = self.context.energy_analysis_result
        self.equipment.blockSignals(True); chosen = self.equipment.currentData(); self.equipment.clear(); self.equipment.addItem(tr("analysis_all_equipment"), None)
        for item in (self.context.equipment_organization.equipment if self.context.equipment_organization else []): self.equipment.addItem(item.name, normalize_equipment_id(item.name))
        self.equipment.setCurrentIndex(max(0, self.equipment.findData(chosen))); self.equipment.blockSignals(False)
        result = self._result()
        while self.chart_layout.count() > 1:
            child = self.chart_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
        if result is None:
            self.metadata.setText(tr("energy_no_data")); return
        q = result.data_quality; self.metadata.setText(tr("energy_metadata", start=result.start or "—", end=result.end or "—", interval=(f"{result.sampling_interval_minutes:g}" if result.sampling_interval_minutes else "—"), equipment=result.summary["equipment_count"], energy=result.summary["energy_points"], temperature=result.summary["temperature_points"], missing=q["missing_ratio"]))
        summary = result.summary
        self.cards["energy"].set_value("—" if summary["total_energy_kwh"] is None else f"{summary['total_energy_kwh']:.1f} kWh")
        self.cards["peak"].set_value("—" if summary["peak_power_kw"] is None else f"{summary['peak_power_kw']:.1f} kW")
        self.cards["cop"].set_value("—" if summary["average_cop"] is None else f"{summary['average_cop']:.2f}")
        self.cards["dt"].set_value("—" if summary["average_delta_t_c"] is None else f"{summary['average_delta_t_c']:.2f} °C")
        for code, title_key, kind in self.CHARTS:
            payload = result.charts.get(code)
            if not payload: continue
            card = QFrame(); card.setObjectName("Card"); box = QVBoxLayout(card); title = QLabel(tr(title_key)); title.setObjectName("CardTitle"); box.addWidget(title)
            unit = QLabel(payload.get("unit", "")); unit.setObjectName("Muted"); box.addWidget(unit)
            chart = TimeSeriesChart(kind); chart.set_payload(payload); box.addWidget(chart); self.chart_layout.insertWidget(self.chart_layout.count()-1, card)
        if result.warnings or q["warnings"]:
            notice = QLabel(tr("energy_quality_warning", warnings=", ".join([*q["warnings"], *result.warnings][:5]))); notice.setObjectName("Muted"); notice.setWordWrap(True); self.chart_layout.insertWidget(0, notice)


class AnalyticsPage(BasePage):
    navigation_requested = pyqtSignal(str, object)

    def __init__(self, context):
        super().__init__(context, "analysis")
        self._analysis_worker = None
        controls = QHBoxLayout()
        self.run_button = QPushButton(); self.run_button.setObjectName("PrimaryButton"); self.run_button.clicked.connect(self.run_diagnosis); controls.addWidget(self.run_button)
        controls.addStretch(1)
        self.filter_label = QLabel(); self.filter_label.setObjectName("Muted"); controls.addWidget(self.filter_label)
        self.equipment_filter = QComboBox(); self.equipment_filter.currentIndexChanged.connect(self.refresh); controls.addWidget(self.equipment_filter)
        self.severity_filter = QComboBox(); self.severity_filter.currentIndexChanged.connect(self.refresh); controls.addWidget(self.severity_filter)
        self.type_filter = QComboBox(); self.type_filter.currentIndexChanged.connect(self.refresh); controls.addWidget(self.type_filter)
        self.export_md = QPushButton(); self.export_html = QPushButton(); self.export_md.clicked.connect(lambda: self.export_report("md")); self.export_html.clicked.connect(lambda: self.export_report("html")); controls.addWidget(self.export_md); controls.addWidget(self.export_html)
        self.layout.addLayout(controls)
        self.progress_panel = AnalysisProgressPanel(); self.progress_panel.hide(); self.layout.addWidget(self.progress_panel)
        self.empty_state = QLabel(); self.empty_state.setObjectName("Muted"); self.empty_state.setAlignment(Qt.AlignCenter); self.empty_state.setWordWrap(True); self.empty_state.setMinimumHeight(160); self.layout.addWidget(self.empty_state)
        self.tabs = QTabWidget(); self.layout.addWidget(self.tabs, 1)
        self.overview = QWidget(); overview_layout = QVBoxLayout(self.overview); overview_layout.setContentsMargins(0, 0, 0, 0); overview_layout.setSpacing(SPACING_MD)
        summary_grid = QGridLayout(); summary_grid.setSpacing(SPACING_MD)
        self.summary_cards = {
            "equipment": Card("analysis_equipment_total"), "normal": Card("analysis_normal_equipment"), "attention": Card("analysis_attention_equipment"),
            "findings": Card("analysis_high_confidence_findings"), "opportunities": Card("analysis_opportunity_total"),
        }
        for index, card in enumerate(self.summary_cards.values()): summary_grid.addWidget(card, 0, index)
        overview_layout.addLayout(summary_grid)
        charts = QGridLayout(); charts.setSpacing(SPACING_MD)
        self.cop_chart = MetricBarChart("", warning_below=2.5)
        self.delta_chart = MetricBarChart("°C", warning_below=3.0)
        self.finding_chart = MetricBarChart("")
        for index, (title_key, chart) in enumerate((("analysis_chart_cop", self.cop_chart), ("analysis_chart_delta_t", self.delta_chart), ("analysis_chart_findings", self.finding_chart))):
            card = QFrame(); card.setObjectName("Card"); box = QVBoxLayout(card); box.setContentsMargins(SPACING_MD, SPACING_MD, SPACING_MD, SPACING_MD)
            title = QLabel(); title.setObjectName("CardTitle"); title.setProperty("title_key", title_key); box.addWidget(title); box.addWidget(chart)
            charts.addWidget(card, 0, index)
        overview_layout.addLayout(charts); overview_layout.addStretch(1)
        self.tabs.addTab(self.overview, "")
        self.kpi_scroll, self.kpi_content, self.kpi_layout = self._scroll_tab(); self.tabs.addTab(self.kpi_scroll, "")
        self.findings_page = QWidget(); findings_box = QVBoxLayout(self.findings_page); findings_box.setContentsMargins(0, 0, 0, 0)
        finding_split = QSplitter(); self.finding_table = QTableWidget(0, 5); self.setup_table(self.finding_table); self.finding_table.itemSelectionChanged.connect(self._show_finding_detail); finding_split.addWidget(self.finding_table)
        detail_frame = QFrame(); detail_frame.setObjectName("FindingDetail"); detail_box = QVBoxLayout(detail_frame); self.finding_detail = QTextEdit(); self.finding_detail.setReadOnly(True); detail_box.addWidget(self.finding_detail, 1)
        self.finding_trend = EngineeringTimeSeriesChart("line"); self.finding_trend.setMinimumHeight(250); self.finding_trend.setVisible(False); detail_box.addWidget(self.finding_trend)
        self.finding_ask_ai = QPushButton(); self.finding_ask_ai.setObjectName("PrimaryButton"); self.finding_ask_ai.clicked.connect(self._ask_about_finding); detail_box.addWidget(self.finding_ask_ai); finding_split.addWidget(detail_frame); finding_split.setSizes([560, 480]); findings_box.addWidget(finding_split); self.tabs.addTab(self.findings_page, "")
        self.passed_scroll, self.passed_content, self.passed_layout = self._scroll_tab(); self.tabs.addTab(self.passed_scroll, "")
        self.opportunities_scroll, self.opportunities_content, self.opportunities_layout = self._scroll_tab(); self.tabs.addTab(self.opportunities_scroll, "")
        self.retranslate_ui()

    @staticmethod
    def _scroll_tab():
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget(); layout = QVBoxLayout(content); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(SPACING_MD); layout.addStretch(1)
        scroll.setWidget(content)
        return scroll, content, layout

    @staticmethod
    def _clear_cards(layout):
        while layout.count() > 1:
            item = layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

    @staticmethod
    def _mean(series):
        try:
            value = series.dropna().mean()
            return None if value != value else float(value)
        except Exception:
            return None

    @staticmethod
    def _card(title: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame(); card.setObjectName("Card")
        box = QVBoxLayout(card); box.setContentsMargins(SPACING_MD, SPACING_MD, SPACING_MD, SPACING_MD); box.setSpacing(SPACING_SM)
        heading = QLabel(title); heading.setObjectName("CardTitle"); heading.setWordWrap(True); box.addWidget(heading)
        return card, box

    def _selected_name(self):
        return self.equipment_filter.currentData() if hasattr(self, "equipment_filter") else None

    @staticmethod
    def _confidence(value):
        return tr("finding_confidence_high") if value >= .8 else tr("finding_confidence_medium") if value >= .6 else tr("finding_confidence_low")

    def _metric_evidence(self, finding):
        evidence = finding.evidence[0].metric_value if finding.evidence else {}
        if not isinstance(evidence, dict): return str(evidence or "—")
        parts = []
        for key, value in evidence.items():
            display = f"{value:.1%}" if "ratio" in key and isinstance(value, (float, int)) else f"{value:.2f}" if isinstance(value, float) else str(value)
            parts.append(f"{tr('analysis_metric_' + key)}: {display}")
        return " · ".join(parts[:5]) or "—"

    def retranslate_ui(self):
        super().retranslate_ui()
        if hasattr(self, "run_button"):
            self.run_button.setText(tr("analysis_running_button") if self._analysis_worker else tr("run_analysis"))
            self.filter_label.setText(tr("analysis_equipment_filter"))
            severity = self.severity_filter.currentData(); self.severity_filter.blockSignals(True); self.severity_filter.clear(); self.severity_filter.addItem(tr("diagnostics_all_severity"), None)
            for value in ("critical", "warning", "info"): self.severity_filter.addItem(tr("finding_severity_" + value), value)
            self.severity_filter.setCurrentIndex(max(0, self.severity_filter.findData(severity))); self.severity_filter.blockSignals(False)
            self.type_filter.setToolTip(tr("diagnostics_finding_type")); self.export_md.setText(tr("diagnostics_export_markdown")); self.export_html.setText(tr("diagnostics_export_html")); self.finding_ask_ai.setText(tr("equipment_ask_ai"))
            self.finding_table.setHorizontalHeaderLabels([tr("equipment"), tr("diagnostics_finding_column"), tr("status"), tr("analysis_period"), tr("analysis_evidence")])
            self.tabs.setTabText(0, tr("analysis_overview")); self.tabs.setTabText(1, tr("analysis_kpi")); self.tabs.setTabText(2, tr("analysis_findings")); self.tabs.setTabText(3, tr("diagnostics_passed_checks")); self.tabs.setTabText(4, tr("analysis_opportunities"))
            for card in self.summary_cards.values(): card.retranslate_ui()
            for title in self.findChildren(QLabel):
                if title.property("title_key"): title.setText(tr(title.property("title_key")))
            self.refresh()

    def run_diagnosis(self):
        if self._analysis_worker is not None:
            return
        if self.context.dataframe is None or not self.context.semantic_result:
            QMessageBox.warning(self, tr("analysis_unavailable"), tr("run_semantics_first")); return
        self.progress_panel.start(); self.run_button.setEnabled(False); self.run_button.setText(tr("analysis_running_button"))
        self._analysis_worker = AnalysisWorker(self.context)
        self._analysis_worker.progress_changed.connect(self.progress_panel.update_progress)
        self._analysis_worker.analysis_finished.connect(self._analysis_finished)
        self._analysis_worker.analysis_failed.connect(self._analysis_failed)
        self._analysis_worker.finished.connect(self._analysis_worker_finished)
        self._analysis_worker.start()

    def _analysis_finished(self, summary: dict):
        self.progress_panel.complete(summary); self.refresh()

    def _analysis_failed(self, stage: str, error: str):
        self.progress_panel.fail(stage, error)

    def _analysis_worker_finished(self):
        worker = self._analysis_worker
        self._analysis_worker = None
        self.run_button.setEnabled(True); self.run_button.setText(tr("run_analysis"))
        if worker:
            worker.deleteLater()

    def refresh(self):
        if not hasattr(self, "tabs"): return
        if self.context.dataframe is None or not self.context.semantic_result:
            self.empty_state.setText(tr("analysis_empty_state")); self.empty_state.show(); self.tabs.hide(); return
        diagnosis = self.context.diagnosis_result
        if diagnosis is None:
            self.empty_state.setText(tr("analysis_ready_state")); self.empty_state.show(); self.tabs.hide(); return
        self.empty_state.hide(); self.tabs.show()
        all_kpis = diagnosis.analytics.equipment_kpis
        selected_before = self._selected_name()
        self.equipment_filter.blockSignals(True); self.equipment_filter.clear(); self.equipment_filter.addItem(tr("analysis_all_equipment"), None)
        for kpi in all_kpis: self.equipment_filter.addItem(kpi.equipment_name, kpi.equipment_name)
        target = next((index for index in range(self.equipment_filter.count()) if self.equipment_filter.itemData(index) == selected_before), 0)
        self.equipment_filter.setCurrentIndex(target); self.equipment_filter.blockSignals(False)
        selected = self._selected_name(); kpis = [item for item in all_kpis if not selected or item.equipment_name == selected]
        names = {item.equipment_id: item.equipment_name for item in all_kpis}
        findings = [item for item in diagnosis.findings if not selected or names.get(item.equipment_id) == selected]
        severity = self.severity_filter.currentData()
        if severity: findings = [item for item in findings if item.severity == severity]
        current_type = self.type_filter.currentData()
        available_types = sorted({item.finding_type for item in diagnosis.findings})
        self.type_filter.blockSignals(True); self.type_filter.clear(); self.type_filter.addItem(tr("diagnostics_all_types"), None)
        for item in available_types: self.type_filter.addItem(tr("finding_" + item + "_title"), item)
        self.type_filter.setCurrentIndex(max(0, self.type_filter.findData(current_type))); self.type_filter.blockSignals(False)
        if current_type: findings = [item for item in findings if item.finding_type == current_type]
        opportunities = [item for item in self.context.opportunities if not selected or names.get(item.equipment_id) == selected]
        high = sum(item.confidence >= .8 for item in findings)
        self.summary_cards["equipment"].set_value(str(len(kpis)), tr("analysis_equipment_total"))
        attention_equipment = {item.equipment_id for item in findings}
        self.summary_cards["normal"].set_value(str(sum(item.status == "available" and item.equipment_id not in attention_equipment for item in kpis)), tr("analysis_normal_equipment"))
        self.summary_cards["attention"].set_value(str(sum(item.status == "available" and item.equipment_id in attention_equipment for item in kpis)), tr("analysis_attention_equipment"))
        self.summary_cards["findings"].set_value(str(high), tr("analysis_high_confidence_findings"))
        self.summary_cards["opportunities"].set_value(str(len(opportunities)), tr("analysis_opportunity_total"))
        self.cop_chart.set_data([(item.equipment_name, item.metric_summary.get("cop", {}).get("mean")) for item in kpis])
        self.delta_chart.set_data([(item.equipment_name, item.metric_summary.get("delta_t_c", {}).get("mean")) for item in kpis])
        finding_counts = {name: 0 for name in [item.equipment_name for item in kpis]}
        for item in findings: finding_counts[names.get(item.equipment_id, "—")] = finding_counts.get(names.get(item.equipment_id, "—"), 0) + 1
        self.finding_chart.set_data(list(finding_counts.items()))
        self._render_kpis(kpis, findings, names)
        self._render_findings(findings, names)
        self._render_passed_checks(diagnosis, selected)
        self._render_opportunities(opportunities, names)
        self.context.cop_status = "Diagnosed" if diagnosis.analytics.available_kpis else "Unavailable"

    def _render_kpis(self, kpis, findings, names):
        self._clear_cards(self.kpi_layout)
        finding_ids = {item.equipment_id for item in findings}
        for kpi in kpis:
            card, box = self._card(kpi.equipment_name)
            state = tr("analysis_available") if kpi.status == "available" else tr("analysis_skipped")
            status = QLabel(f"{tr('status')}: {state} · {tr('analysis_needs_attention') if kpi.equipment_id in finding_ids else tr('analysis_normal')}"); status.setObjectName("Muted"); box.addWidget(status)
            grid = QGridLayout(); grid.setHorizontalSpacing(SPACING_LG); grid.setVerticalSpacing(SPACING_SM)
            metric = kpi.metric_summary
            values = (
                ("analysis_supply_temperature", metric.get("supply_temp_c", {}).get("mean"), "°C"),
                ("analysis_return_temperature", metric.get("return_temp_c", {}).get("mean"), "°C"),
                ("analysis_delta_t", metric.get("delta_t_c", {}).get("mean"), "°C"),
                ("analysis_flow", metric.get("flow_lps", {}).get("mean"), " L/s"),
                ("analysis_input_power", metric.get("power_kw", {}).get("mean"), " kW"),
                ("analysis_thermal_load", metric.get("thermal_load_kw", {}).get("mean"), " kW"),
                ("analysis_cop", metric.get("cop", {}).get("mean"), ""),
            )
            for index, (key, value, suffix) in enumerate(values):
                label = QLabel(tr(key)); label.setObjectName("Muted"); number = QLabel("N/A" if value is None else f"{value:.2f}{suffix}"); number.setObjectName("CardValue")
                grid.addWidget(label, index // 2 * 2, index % 2); grid.addWidget(number, index // 2 * 2 + 1, index % 2)
            box.addLayout(grid)
            if kpi.status != "available":
                reason = QLabel(f"{tr('analysis_reason')}: {reason_text(kpi.reason)}"); reason.setObjectName("Muted"); reason.setWordWrap(True); box.addWidget(reason)
            self.kpi_layout.insertWidget(self.kpi_layout.count() - 1, card)
        if not kpis: self.kpi_layout.insertWidget(0, QLabel(tr("analysis_no_kpi")))

    def _render_findings(self, findings, names):
        self._current_findings = list(findings); self._finding_names = names
        self.finding_table.setRowCount(len(findings))
        for row, item in enumerate(findings):
            period = item.affected_period or {}
            values = [names.get(item.equipment_id, item.equipment_id or "—"), finding_text(item)[0], tr("finding_severity_" + item.severity), f"{period.get('start') or '—'} — {period.get('end') or '—'}", finding_evidence_summary(item)]
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value); cell.setData(Qt.UserRole, item.finding_id); self.finding_table.setItem(row, column, cell)
        target = self.context.selected_finding_id
        target_row = next((row for row, item in enumerate(findings) if item.finding_id == target), 0)
        if findings: self.finding_table.selectRow(target_row)
        else: self.finding_detail.setPlainText(tr("analysis_no_findings") + "\n\n" + tr("dashboard_not_health_claim"))

    def _show_finding_detail(self):
        row = self.finding_table.currentRow()
        if not 0 <= row < len(getattr(self, "_current_findings", [])): return
        item = self._current_findings[row]; self.context.selected_finding_id = item.finding_id
        user = next((value for value in getattr(self.context, "user_interpretations", []) if value.finding_id == item.finding_id), None)
        title, description = (user.problem, user.explanation) if user else finding_text(item)
        actions = "\n".join(f"  {index + 1}. {action}" for index, action in enumerate(user.actions)) if user else tr("analysis_no_opportunities")
        possible = tr("finding_" + item.finding_type + "_causes")
        if possible == "finding_" + item.finding_type + "_causes": possible = user.explanation if user else tr("analysis_possible_causes") + ": N/A"
        verification = user.expected_effect if user else "Re-run the same KPI and rule after site inspection."
        equipment_name = self._finding_names.get(item.equipment_id, item.equipment_id)
        trend_payload = self._finding_trend_payload(item, equipment_name)
        self.finding_trend.set_payload(trend_payload); self.finding_trend.setVisible(bool(trend_payload))
        result = getattr(self.context, "energy_analysis_view_result", None) or self.context.energy_analysis_result
        if trend_payload and result:
            supporting = tr("diagnostics_measured_trend") + "\n" + tr(
                "energy_chart_scope", start=EngineeringTimeSeriesChart._scope_time(result.start),
                end=EngineeringTimeSeriesChart._scope_time(result.end),
                resolution=EnergyAnalysisPage._resolution_label(result),
                equipment=equipment_name or tr("analysis_all_equipment"),
            )
        else:
            supporting = tr("energy_no_data")
        text = (
            f"{title}\n\n{tr('diagnostics_what_happened')}\n{description}\n\n"
            f"{tr('diagnostics_project_evidence')}\n{finding_evidence_summary(item)}\n\n"
            f"{tr('diagnostics_why')}\n{possible}\n\n{tr('diagnostics_inspect')}\n{actions}\n\n"
            f"{tr('diagnostics_energy_impact')}\n{tr('diagnostics_energy_impact_unavailable')}\n\n"
            f"{tr('diagnostics_supporting_trends')}\n{supporting}\n{', '.join(item.source_metrics) or 'N/A'}\n\n"
            f"{tr('diagnostics_verification')}\n{verification}\n\n{tr('diagnostics_reference')}\n{tr('knowledge_base')}"
        )
        self.finding_detail.setPlainText(text)

    def _finding_trend_payload(self, finding, equipment_name: str | None) -> dict:
        result = getattr(self.context, "energy_analysis_view_result", None) or self.context.energy_analysis_result
        if not result:
            return {}
        chart_code, y_key = {
            "low_heat_source_cop": ("cop_trend", "energy_axis_cop"),
            "low_chilled_water_delta_t": ("delta_t_trend", "energy_axis_delta_t"),
            "off_hour_operation": ("power_trend", "energy_axis_power"),
            "low_load_high_power": ("power_trend", "energy_axis_power"),
            "frequent_start_stop": ("power_trend", "energy_axis_power"),
            "parallel_heat_source_operation": ("power_trend", "energy_axis_power"),
        }.get(finding.finding_type, ("", ""))
        payload = dict(result.charts.get(chart_code) or {})
        if not payload:
            return {}
        series = list(payload.get("series", []))
        selected = [item for item in series if equipment_name and str(item.get("name", "")).startswith(equipment_name)]
        if selected:
            payload["series"] = selected
        elif equipment_name and series:
            # A building-total or different-equipment trend is not valid
            # supporting evidence for this equipment-specific finding.
            return {}
        payload["x_label"] = tr("energy_axis_time")
        payload["y_label"] = tr(y_key)
        return payload

    def _render_passed_checks(self, diagnosis, selected):
        self._clear_cards(self.passed_layout)
        rows = [item for item in passed_checks(diagnosis) if not selected or item.equipment_name == selected]
        for item in rows:
            card, box = self._card(f"✓ {item.equipment_name} · {item.title}")
            note = QLabel(tr("diagnostics_passed_note")); note.setObjectName("Muted"); note.setWordWrap(True); box.addWidget(note)
            self.passed_layout.insertWidget(self.passed_layout.count() - 1, card)
        if not rows: self.passed_layout.insertWidget(0, QLabel(tr("analysis_no_kpi")))

    def _ask_about_finding(self):
        row = self.finding_table.currentRow()
        if not 0 <= row < len(getattr(self, "_current_findings", [])): return
        item = self._current_findings[row]; equipment = self._finding_names.get(item.equipment_id, item.equipment_id)
        self.navigation_requested.emit("agent", {"equipment_id": equipment, "finding_id": item.finding_id, "prompt": f"Why does {equipment} have this finding, what project evidence supports it, and what should I inspect first?"})

    def export_report(self, extension: str):
        if not self.context.current_project or not self.context.energy_analysis_result or not self.context.diagnosis_result:
            QMessageBox.warning(self, tr("analysis_unavailable"), tr("analysis_ready_state")); return
        path, _ = QFileDialog.getSaveFileName(self, tr("diagnostics_export_markdown" if extension == "md" else "diagnostics_export_html"), f"buildingai-analysis.{extension}", "Markdown (*.md)" if extension == "md" else "HTML (*.html)")
        if not path: return
        try:
            body = AnalysisReportService.markdown(self.context.current_project, self.context.energy_analysis_result, self.context.diagnosis_result, self.context.opportunities) if extension == "md" else AnalysisReportService.html(self.context.current_project, self.context.energy_analysis_result, self.context.diagnosis_result, self.context.opportunities)
            with open(path, "w", encoding="utf-8") as handle: handle.write(body)
        except OSError:
            QMessageBox.warning(self, tr("report_failed"), tr("report_failed")); return
        QMessageBox.information(self, tr("report_saved"), tr("report_saved"))

    def apply_global_context(self):
        if hasattr(self, "equipment_filter"):
            index = self.equipment_filter.findData(self.context.selected_equipment_id)
            if index >= 0: self.equipment_filter.setCurrentIndex(index)
        self.refresh()

    def _render_opportunities(self, opportunities, names):
        self._clear_cards(self.opportunities_layout)
        if not opportunities:
            self.opportunities_layout.insertWidget(0, QLabel(tr("analysis_no_opportunities"))); return
        finding_to_user = {item.finding_id: item for item in getattr(self.context, "user_interpretations", [])}
        for item in opportunities:
            user = next((finding_to_user[x] for x in item.related_finding_ids if x in finding_to_user), None)
            title, description = (user.problem, user.explanation) if user else opportunity_text(item)
            card, box = self._card(f"{names.get(item.equipment_id, item.equipment_id or '—')} · {title}")
            if user:
                meta = QLabel(f"{tr('analysis_priority')}: {user.priority} · {tr('recommendation_difficulty')}: {user.implementation_difficulty}"); meta.setObjectName("Muted"); box.addWidget(meta)
                body = QLabel(tr("diagnostics_inspect") + "\n" + "\n".join(f"{index + 1}. {action}" for index, action in enumerate(user.actions))); body.setWordWrap(True); box.addWidget(body)
                impact = QLabel(f"{tr('recommendation_effect')}: {user.expected_effect}\n{tr('diagnostics_verification')}: {tr('recommendation_verify_kpi')}\n{tr('diagnostics_energy_impact')}: {tr('diagnostics_energy_impact_unavailable')}"); impact.setObjectName("Muted"); impact.setWordWrap(True); box.addWidget(impact)
            else:
                meta = QLabel(f"{tr('analysis_priority')}: {opportunity_priority_text(item)} · {tr('analysis_confidence')}: {self._confidence(item.confidence)}"); meta.setObjectName("Muted"); box.addWidget(meta)
                body = QLabel(description); body.setWordWrap(True); box.addWidget(body)
            self.opportunities_layout.insertWidget(self.opportunities_layout.count() - 1, card)


class AgentPage(BasePage):
    """Chat presentation layer for the existing read-only Agent tool registry."""

    message_submitted = pyqtSignal(str)
    TOOL_IDS = ("get_project_summary", "get_semantic_mapping", "get_point_timeseries", "get_analysis_results", "get_equipment_kpis", "get_equipment_summary", "get_diagnostic_findings", "get_energy_opportunities", "get_energy_summary", "get_energy_timeseries", "get_temperature_summary", "list_projects")

    def __init__(self, context):
        super().__init__(context, "agent")
        self._agent_worker = None
        self._tool_calls: dict[str, ToolCallWidget] = {}
        self._tool_call_sequence = 0
        self._welcome_added = False
        self._user_message_count = 0
        self._active_project_id = None
        self._conversation_id = "gui-no-project"
        self._current_process = None
        self._request_started = 0.0

        header = QFrame(); header.setObjectName("Card")
        header_layout = QHBoxLayout(header); header_layout.setContentsMargins(SPACING_MD, SPACING_SM, SPACING_MD, SPACING_SM)
        self.project_status = QLabel(); self.project_status.setObjectName("Muted")
        self.focus_status = QLabel(); self.focus_status.setObjectName("Muted")
        self.clear_focus_button = QPushButton(); self.clear_focus_button.setObjectName("TextButton"); self.clear_focus_button.clicked.connect(self.clear_focus)
        self.model_status = QLabel(); self.model_status.setObjectName("Muted")
        self.agent_status = QLabel(); self.agent_status.setObjectName("Muted")
        header_layout.addWidget(self.project_status); header_layout.addWidget(self.focus_status); header_layout.addWidget(self.clear_focus_button); header_layout.addStretch(1); header_layout.addWidget(self.model_status); header_layout.addWidget(self.agent_status)
        self.layout.addWidget(header)

        body = QHBoxLayout(); body.setSpacing(SPACING_MD)
        chat_card = QFrame(); chat_card.setObjectName("Card")
        chat_layout = QVBoxLayout(chat_card); chat_layout.setContentsMargins(SPACING_SM, SPACING_SM, SPACING_SM, SPACING_SM)
        self.setup_notice = QFrame(); self.setup_notice.setObjectName("AgentSetupNotice")
        setup_layout = QHBoxLayout(self.setup_notice); setup_layout.setContentsMargins(SPACING_MD, SPACING_SM, SPACING_MD, SPACING_SM)
        self.setup_text = QLabel(); self.setup_text.setWordWrap(True)
        self.open_settings_button = QPushButton(); self.open_settings_button.clicked.connect(self.open_settings)
        setup_layout.addWidget(self.setup_text, 1); setup_layout.addWidget(self.open_settings_button)
        chat_layout.addWidget(self.setup_notice)
        self.chat_scroll = QScrollArea(); self.chat_scroll.setWidgetResizable(True); self.chat_scroll.setFrameShape(QFrame.NoFrame)
        self.transcript = ChatTranscript(); self.chat_scroll.setWidget(self.transcript); chat_layout.addWidget(self.chat_scroll, 1)
        body.addWidget(chat_card, 1)

        tools_card = QFrame(); tools_card.setObjectName("Card"); tools_card.setFixedWidth(230)
        tools_layout = QVBoxLayout(tools_card); tools_layout.setContentsMargins(SPACING_MD, SPACING_MD, SPACING_MD, SPACING_MD)
        self.tools_title = QLabel(); self.tools_title.setObjectName("CardTitle")
        self.tools_list = QLabel(); self.tools_list.setWordWrap(True); self.tools_list.setObjectName("Muted")
        tools_layout.addWidget(self.tools_title); tools_layout.addWidget(self.tools_list); tools_layout.addStretch(1)
        body.addWidget(tools_card)
        self.layout.addLayout(body, 1)

        input_row = QHBoxLayout(); input_row.setSpacing(SPACING_SM)
        self.input = ChatInput(); self.input.setObjectName("ChatInput"); self.input.setFixedHeight(62); self.input.submitted.connect(self.submit_message); self.input.textChanged.connect(self._update_send_state)
        self.send_button = QPushButton(); self.send_button.setObjectName("PrimaryButton"); self.send_button.clicked.connect(self.submit_message)
        input_row.addWidget(self.input, 1); input_row.addWidget(self.send_button, alignment=Qt.AlignBottom)
        self.layout.addLayout(input_row)
        self.retranslate_ui()

    def retranslate_ui(self):
        super().retranslate_ui()
        if not hasattr(self, "input"):
            return
        self.input.setPlaceholderText(tr("agent_input_placeholder"))
        self.send_button.setText(tr("send"))
        self.open_settings_button.setText(tr("open_settings"))
        self.setup_text.setText(tr("agent_llm_not_configured"))
        self.tools_title.setText(tr("agent_context_title"))
        self.tools_list.setText(tr("agent_context_text"))
        self._render_suggestions()
        self.refresh(); self._set_focus(getattr(self, '_focus', None))

    def refresh(self):
        if not hasattr(self, "input"):
            return
        project = self.context.current_project
        project_id = project.project_id if project else None
        if project_id != self._active_project_id:
            self._active_project_id = project_id
            self._conversation_id = f"gui-{project_id or 'no-project'}-{int(time.monotonic() * 1000)}"
            self._set_focus(None)
        self.project_status.setText(f"{tr('current_project')}: {project.name if project else tr('no_project')}")
        provider = self.context.llm_manager.get_provider()
        if provider.is_configured:
            ok, _ = provider.test_connection(timeout=0.35)
            state = tr("connected") if ok else tr("connection_error")
            color = "#16A34A" if ok else "#DC2626"
            self.model_status.setText(f"{tr('model')}: {provider.display_name}  <span style='color:{color}'>●</span> {state}")
            self.setup_notice.setVisible(False)
            self.set_input_enabled(True)
            self._ensure_welcome()
        else:
            self.model_status.setText(f"{tr('llm')}: {tr('not_configured')}  <span style='color:#D97706'>●</span>")
            self.setup_notice.setVisible(True)
            self.set_input_enabled(False)
        self._update_send_state()

    def _set_focus(self, equipment: str | None) -> None:
        self._focus = equipment
        if equipment:
            self.focus_status.setText(f"{tr('agent_focus')}: {equipment}")
            self.clear_focus_button.setText("× " + tr("agent_clear_focus")); self.clear_focus_button.setVisible(True)
        else:
            self.focus_status.setText(""); self.clear_focus_button.setVisible(False)

    def _refresh_focus(self) -> None:
        project = self.context.current_project
        if not project:
            self._set_focus(None); return
        value = MemoryStore(self.context.database).get(project.project_id, self._conversation_id, 'focus', 'equipment')
        self._set_focus(value.get('equipment_id') if value else None)

    def clear_focus(self) -> None:
        project = self.context.current_project
        if project:
            MemoryStore(self.context.database).delete(project.project_id, self._conversation_id, 'focus', 'equipment')
        self._set_focus(None)

    def _ensure_welcome(self) -> None:
        if self._welcome_added:
            return
        self._welcome_added = True
        self.append_assistant_message("", translation_key="agent_greeting")
        self._render_suggestions()

    def _render_suggestions(self) -> None:
        if not hasattr(self, "transcript"):
            return
        if hasattr(self, "suggestions"):
            self.suggestion_title.setText(tr("recommended_questions"))
            for button in self.suggestion_buttons:
                button.setText(tr(button.property("translation_key")))
            self.suggestions.setVisible(self._user_message_count == 0)
            return
        self.suggestions = QFrame(); self.suggestions.setObjectName("AgentSuggestions")
        layout = QVBoxLayout(self.suggestions); layout.setContentsMargins(SPACING_SM, SPACING_SM, SPACING_SM, SPACING_SM)
        self.suggestion_title = QLabel(tr("recommended_questions")); self.suggestion_title.setObjectName("CardTitle"); layout.addWidget(self.suggestion_title)
        self.suggestion_buttons = []
        keys = ["agent_question_summary"]
        diagnosis = getattr(self.context, 'diagnosis_result', None)
        if diagnosis and diagnosis.analytics.available_kpis: keys.append("agent_question_cop")
        if diagnosis and diagnosis.findings: keys.append("agent_question_findings")
        if not diagnosis: keys.append("agent_question_review")
        for key in keys:
            button = QPushButton(tr(key)); button.setObjectName("SuggestionButton"); button.setProperty("translation_key", key)
            button.clicked.connect(lambda checked=False, text_key=key: self.fill_suggestion(text_key))
            self.suggestion_buttons.append(button); layout.addWidget(button)
        self.transcript.append(self.suggestions)

    def fill_suggestion(self, text_key: str) -> None:
        if self.input.isEnabled():
            self.input.setPlainText(tr(text_key)); self.input.setFocus()

    def set_context_focus(self, equipment: str | None = None, finding_id: str | None = None, prompt: str | None = None) -> None:
        """Receive bounded page context without copying another page's data."""
        self.refresh()
        project = self.context.current_project
        if project and equipment:
            MemoryStore(self.context.database).put(
                project.project_id, self._conversation_id, "focus", "equipment",
                {"equipment_id": equipment, "source": "ui_context"},
            )
        self.context.selected_equipment_id = equipment or self.context.selected_equipment_id
        self.context.selected_finding_id = finding_id or self.context.selected_finding_id
        self._set_focus(equipment)
        if prompt:
            self.input.setPlainText(prompt)
            if self.input.isEnabled(): self.input.setFocus()

    def append_user_message(self, text: str) -> ChatMessage:
        self._user_message_count += 1
        if hasattr(self, "suggestions"):
            self.suggestions.setVisible(False)
        message = ChatMessage("user", text)
        self.transcript.append(message); self.transcript.scroll_to_bottom(self.chat_scroll)
        return message

    def append_assistant_message(self, text: str, *, translation_key: str | None = None) -> ChatMessage:
        message = ChatMessage("assistant", text, translation_key=translation_key)
        self.transcript.append(message); self.transcript.scroll_to_bottom(self.chat_scroll)
        return message

    def show_tool_call(self, tool_id: str, status: str = "RUNNING", detail: str = "") -> str:
        self._tool_call_sequence += 1
        handle = f"tool_call_{self._tool_call_sequence}"
        widget = ToolCallWidget(tool_id, status, detail)
        self._tool_calls[handle] = widget
        self.transcript.append(widget); self.transcript.scroll_to_bottom(self.chat_scroll)
        return handle

    def update_tool_call(self, handle: str, status: str, detail: str = "") -> None:
        if handle in self._tool_calls:
            self._tool_calls[handle].update_status(status, detail)

    def set_agent_status(self, status_key: str = "") -> None:
        self.agent_status.setText(tr(status_key) if status_key else "")

    def set_input_enabled(self, enabled: bool) -> None:
        self.input.setEnabled(enabled)
        self._update_send_state()

    def _update_send_state(self) -> None:
        self.send_button.setEnabled(self.input.isEnabled() and bool(self.input.toPlainText().strip()))

    def clear_chat(self) -> None:
        self.transcript.clear(); self._tool_calls.clear(); self._tool_call_sequence = 0; self._welcome_added = False; self._user_message_count = 0
        if hasattr(self, "suggestions"):
            del self.suggestions
        self._ensure_welcome()

    def set_message_handler(self, handler) -> None:
        """Register a future controller callback without coupling this UI to Agent logic."""
        self._message_handler = handler

    def submit_message(self) -> None:
        text = self.input.toPlainText().strip()
        if not text or not self.input.isEnabled():
            return
        self.append_user_message(text); self.input.clear(); self.message_submitted.emit(text)
        self.set_agent_status("agent_analyzing")
        self.set_input_enabled(False)
        project = self.context.current_project
        project_id = project.project_id if project else None
        runtime = create_agent_runtime(self.context)
        route = runtime.route(text); plan = runtime.plan(route, project_id) if project_id else None
        self._current_process = AgentProcessCard(route, plan); self.transcript.append(self._current_process); self.transcript.scroll_to_bottom(self.chat_scroll)
        self._request_started = time.monotonic()
        self._agent_worker = AgentWorker(lambda message: create_agent_runtime(self.context).run(message, project_id, self._conversation_id), text)
        self._agent_worker.completed.connect(self._agent_completed)
        self._agent_worker.failed.connect(self._agent_failed)
        self._agent_worker.start()

    def _agent_tool_event(self, name: str, state: str) -> None:
        handle = self._tool_handles.get(name)
        if handle is None:
            handle = self.show_tool_call(name, state); self._tool_handles[name] = handle
        else:
            self.update_tool_call(handle, state)

    def _project_evidence_lines(self, trace: dict) -> list[str]:
        project = self.context.current_project
        if not project:
            return []
        lines = [f"{tr('current_project')}: {project.name}"]
        diagnosis = self.context.diagnosis_result
        if diagnosis:
            focus = getattr(self, '_focus', None)
            kpis = [item for item in diagnosis.analytics.equipment_kpis if not focus or item.equipment_name == focus]
            for item in kpis[:1]:
                metrics = item.metric_summary; cop = metrics.get('cop', {}).get('mean'); delta_t = metrics.get('delta_t_c', {}).get('mean')
                if isinstance(cop, (int, float)): lines.append(f"{item.equipment_name} · COP: {cop:.2f}")
                if isinstance(delta_t, (int, float)): lines.append(f"{item.equipment_name} · ΔT: {delta_t:.2f} °C")
                # Low-ΔT occurrence belongs to the formal diagnostic finding,
                # while the KPI summary holds the mean. Keep both project-data
                # facts visible without deriving a finding from RAG content.
                low_delta = next((finding for finding in diagnosis.findings if finding.equipment_id == item.equipment_id and finding.finding_type == 'low_chilled_water_delta_t'), None)
                if low_delta and low_delta.valid_sample_count:
                    lines.append(f"{tr('analysis_metric_low_delta_t_samples')}: {low_delta.occurrence_count} / {low_delta.valid_sample_count}")
            findings = diagnosis.findings
            if findings: lines.append(f"{len(findings)} {tr('analysis_findings').lower()}")
        return lines

    def _append_context_actions(self, trace: dict) -> None:
        tools = {item.get('tool') for item in trace.get('tool_calls', [])}
        actions = []
        window = self.window()
        if getattr(self, '_focus', None): actions.append((tr('equipment'), "equipment"))
        if {'get_diagnostic_findings', 'get_analysis_results'} & tools: actions.append((tr('analysis'), "analysis"))
        if {'get_energy_summary', 'get_energy_timeseries'} & tools: actions.append((tr('energy_analysis'), "energy_analysis"))
        if not actions or not hasattr(window, 'navigate_to'): return
        row = QHBoxLayout(); row.setSpacing(SPACING_SM)
        for label, page_key in actions[:3]:
            button = QPushButton(label); button.setObjectName('TextButton'); button.clicked.connect(lambda checked=False, key=page_key: window.navigate_to(key, {"equipment_id": getattr(self, '_focus', None)})); row.addWidget(button)
        row.addStretch(1); holder = QFrame(); holder.setLayout(row); self.transcript.append(holder)

    def _agent_completed(self, response) -> None:
        trace = TraceStore(self.context.database).get(response.trace_id)
        elapsed_ms = (time.monotonic() - self._request_started) * 1000
        if self._current_process and trace:
            self._current_process.complete(trace, elapsed_ms)
        self._refresh_focus()
        self.append_assistant_message(response.answer)
        if trace:
            evidence = self._project_evidence_lines(trace)
            if evidence and response.grounded: self.transcript.append(AgentEvidenceCard(evidence))
            if response.sources: self.transcript.append(AgentSourcesCard(response.sources, trace))
            self._append_context_actions(trace)
        self.transcript.scroll_to_bottom(self.chat_scroll)
        self.set_agent_status(""); self.set_input_enabled(True); self._agent_worker = None; self._current_process = None

    def _agent_failed(self) -> None:
        self.append_assistant_message("", translation_key="agent_backend_failed"); self.set_agent_status(""); self.set_input_enabled(True); self._agent_worker = None

    def open_settings(self) -> None:
        window = self.window()
        if hasattr(window, "change_page"):
            target = next((index for index, (_, key, _) in enumerate(window.NAVIGATION) if key == "settings"), None)
            if target is not None:
                window.change_page(target)


class SettingsPage(BasePage):
    PROVIDERS = (("not_configured", "disabled"), ("local_llm", "local_llm"), ("openai_compatible", "openai_compatible"), ("custom", "custom"))
    def __init__(self, context):
        super().__init__(context, "settings")
        self.form = QFormLayout(); self.form.setSpacing(SPACING_MD); self._local_models = []
        self.provider = QComboBox(); self.model = QComboBox(); self.model.setEditable(True); self.model.addItem(context.settings.model) if context.settings.model else None
        self.detection_status = QLabel(); self.detection_status.setWordWrap(True); self.detection_status.setObjectName("Muted")
        self.api_base = QLineEdit(context.settings.api_base); self.api_key = QLineEdit(context.settings.api_key); self.api_key.setEchoMode(QLineEdit.Password)
        self.ollama_url = QLineEdit(context.settings.ollama_url); self.model_path = QLineEdit(context.settings.local_model_path); self.device = QComboBox(); self.device.addItems(["auto", "cpu", "cuda", "mps"]); self.device.setCurrentText(context.settings.local_device)
        self.language = QComboBox(); self.language.addItem("English", "en_US"); self.language.addItem("中文", "zh_CN"); self.language.setCurrentIndex(1 if context.settings.language == "zh_CN" else 0)
        self.agent_mode = QComboBox(); self.agent_mode.addItem(tr("agent_mode_single"), "single"); self.agent_mode.addItem(tr("agent_mode_multi"), "multi"); self.agent_mode.setCurrentIndex(max(0, self.agent_mode.findData(context.settings.agent_mode)))
        self.data_dir = QLineEdit(str(context.settings.data_dir)); self.data_dir.setReadOnly(True)
        self.fields = [("llm_provider", self.provider), ("detection_status", self.detection_status), ("model", self.model), ("api_base", self.api_base), ("api_key", self.api_key), ("model_path", self.model_path), ("device", self.device), ("agent_mode", self.agent_mode), ("language", self.language), ("data_directory", self.data_dir)]
        self.form_labels: dict[QWidget, QLabel] = {}
        for key, widget in self.fields:
            label = QLabel(); self.form_labels[widget] = label; self.form.addRow(label, widget)
        self.ollama_label = QLabel(); self.form.addRow(self.ollama_label, self.ollama_url); self.layout.addLayout(self.form)
        buttons = QHBoxLayout(); self.redetect_button = QPushButton(); self.advanced_button = QPushButton(); self.advanced_button.setCheckable(True); self.test_button = QPushButton(); self.save_button = QPushButton(); self.save_button.setObjectName("PrimaryButton")
        self.redetect_button.clicked.connect(self.detect_local_llm); self.advanced_button.toggled.connect(lambda _: self.update_provider_fields(detect=False)); self.test_button.clicked.connect(self.test_connection); self.save_button.clicked.connect(self.save)
        buttons.addWidget(self.redetect_button); buttons.addWidget(self.advanced_button); buttons.addWidget(self.test_button); buttons.addWidget(self.save_button); buttons.addStretch(1); self.layout.addLayout(buttons); self.connection_status = QLabel(); self.connection_status.setObjectName("Muted"); self.layout.addWidget(self.connection_status); self.layout.addStretch(1)
        self.provider.currentIndexChanged.connect(self.update_provider_fields); self.retranslate_ui(); self.update_provider_fields()

    def retranslate_ui(self):
        super().retranslate_ui()
        if not hasattr(self, "provider"): return
        current = self.provider.currentData() or self.context.settings.provider; self.provider.blockSignals(True); self.provider.clear()
        for provider_id, text_key in self.PROVIDERS: self.provider.addItem(tr(text_key), provider_id)
        self.provider.setCurrentIndex(max(0, self.provider.findData(current))); self.provider.blockSignals(False)
        mode = self.agent_mode.currentData() or self.context.settings.agent_mode; self.agent_mode.blockSignals(True); self.agent_mode.clear()
        self.agent_mode.addItem(tr("agent_mode_single"), "single"); self.agent_mode.addItem(tr("agent_mode_multi"), "multi")
        self.agent_mode.setCurrentIndex(max(0, self.agent_mode.findData(mode))); self.agent_mode.blockSignals(False)
        for key, widget in self.fields: self.form_labels[widget].setText(tr(key))
        self.ollama_label.setText(tr("ollama_url"))
        self.redetect_button.setText(tr("redetect")); self.advanced_button.setText(tr("advanced_settings")); self.test_button.setText(tr("test_connection")); self.save_button.setText(tr("save_settings")); self.update_provider_fields(detect=False)

    def update_provider_fields(self, *_: object, detect: bool = True):
        provider = self.provider.currentData(); local = provider == "local_llm"; remote = provider in {"openai_compatible", "custom"}
        if local and detect:
            self.detect_local_llm()
        self.model.setEditable(not local or self.advanced_button.isChecked())
        for widget in (self.model,): widget.setVisible(provider != "not_configured")
        self.detection_status.setVisible(local)
        for widget in (self.api_base, self.api_key): widget.setVisible(remote)
        show_advanced = local and (self.advanced_button.isChecked() or not self._local_models)
        for widget in (self.ollama_url, self.model_path, self.device): widget.setVisible(show_advanced)
        self.redetect_button.setVisible(local); self.advanced_button.setVisible(local)
        for row, (_, widget) in enumerate(self.fields):
            self.form_labels[widget].setVisible(widget.isVisible())
        self.ollama_label.setVisible(self.ollama_url.isVisible())

    def _apply_to_context(self):
        settings = self.context.settings; settings.provider = self.provider.currentData(); settings.model = self.model.currentText().strip(); settings.api_base = self.api_base.text().strip().rstrip("/"); settings.api_key = self.api_key.text().strip(); settings.ollama_url = self.ollama_url.text().strip().rstrip("/"); settings.local_model_path = self.model_path.text().strip(); settings.local_device = self.device.currentText(); settings.agent_mode = self.agent_mode.currentData(); settings.language = self.language.currentData(); settings.__post_init__(); self.context.reload_llm()
        window = self.window()
        if hasattr(window, "update_status"):
            window.update_status()

    def detect_local_llm(self):
        if self.provider.currentData() != "local_llm":
            return
        self._local_models = discover_local_models(self.context.settings)
        current = self.model.currentText().strip()
        self.model.blockSignals(True); self.model.clear()
        for item in self._local_models:
            self.model.addItem(item.model, item)
        if self._local_models:
            index = next((i for i, item in enumerate(self._local_models) if item.model == current), 0)
            self.model.setCurrentIndex(index)
            selected = self._local_models[index]
            self.ollama_url.setText(selected.endpoint)
            # Make the detected model usable immediately.  Persistence remains an
            # explicit action through Save Settings, so a cancelled change does
            # not overwrite the user's local preference file.
            apply_detected_local_model(self.context.settings, selected)
            self.context.reload_llm()
            window = self.window()
            if hasattr(window, "update_status"):
                window.update_status()
            self.detection_status.setStyleSheet("color: #16A34A;")
            self.detection_status.setText(f"● {tr('local_llm_ready', model=selected.model)}")
        else:
            if current:
                self.model.addItem(current)
            self.detection_status.setStyleSheet("color: #D97706;")
            self.detection_status.setText(f"● {tr('local_llm_not_found')}")
        self.model.blockSignals(False)
        self.update_provider_fields(detect=False)

    def test_connection(self):
        if self.provider.currentData() == "local_llm" and not self._local_models:
            self.detect_local_llm()
        self._apply_to_context(); provider = self.context.llm_manager.get_provider(); ok, message = provider.test_connection()
        if ok and self.provider.currentData() == "local_llm":
            try:
                reply = provider.generate("Reply with exactly: ready", temperature=0).strip()
                ok = bool(reply)
                message = f"{tr('local_llm_connected')}: {reply[:80]}" if ok else tr("connection_error")
            except Exception as exc:
                ok, message = False, str(exc)
        self.connection_status.setStyleSheet(f"color: {'#16A34A' if ok else '#D97706'};"); self.connection_status.setText("● " + message); QMessageBox.information(self, tr("test_result"), message)

    def save(self):
        self._apply_to_context(); self.context.settings.save(); LanguageManager.instance().set_language(self.context.settings.language); self.connection_status.setText(tr("settings_saved")); QMessageBox.information(self, tr("settings"), tr("settings_saved"))
