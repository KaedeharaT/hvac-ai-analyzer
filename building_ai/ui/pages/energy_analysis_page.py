"""Interactive, data-driven Energy Analysis page.

This page contains presentation only.  All calculations and capability
decisions remain in :mod:`building_ai.services.energy_analysis_service`.
"""
from __future__ import annotations

import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from building_ai.core.equipment_identity import normalize_equipment_id
from building_ai.i18n import LanguageManager, tr
from building_ai.ui.theme import SPACING_LG, SPACING_MD, SPACING_SM


class MetricCard(QFrame):
    def __init__(self, title_key: str):
        super().__init__(); self.title_key = title_key; self.setObjectName("Card")
        layout = QVBoxLayout(self); layout.setContentsMargins(SPACING_MD, SPACING_SM, SPACING_MD, SPACING_SM)
        self.value = QLabel("—"); self.value.setObjectName("MetricValue")
        self.title = QLabel(); self.title.setObjectName("Muted")
        layout.addWidget(self.value); layout.addWidget(self.title); self.retranslate_ui()

    def set_value(self, value: str) -> None:
        self.value.setText(value)

    def retranslate_ui(self) -> None:
        self.title.setText(tr(self.title_key))


class TimeSeriesChart(QWidget):
    """Qt renderer for measured service payloads only; no demo values."""
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


class EnergyAnalysisPage(QWidget):
    """Interactive real-data energy view with scoped range/device/aggregation."""
    CHARTS = (
        ("energy_trend", "energy_chart_energy", "line"), ("power_trend", "energy_chart_power", "line"),
        ("temperature_trend", "energy_chart_temperature", "line"), ("delta_t_trend", "energy_chart_delta_t", "line"),
        ("cop_trend", "energy_chart_cop", "line"), ("daily_load_profile", "energy_chart_profile", "line"),
        ("load_heatmap", "energy_chart_heatmap", "heatmap"), ("weather_correlation", "energy_chart_weather", "scatter"),
        ("equipment_ranking", "energy_chart_ranking", "bar"), ("period_comparison", "energy_chart_period", "bar"),
    )

    def __init__(self, context):
        super().__init__(); self.context = context; self.i18n = LanguageManager.instance(); self.i18n.language_changed.connect(self.retranslate_ui)
        self.layout = QVBoxLayout(self); self.layout.setContentsMargins(SPACING_LG, SPACING_LG, SPACING_LG, SPACING_LG); self.layout.setSpacing(SPACING_MD)
        self.heading = QLabel(); self.heading.setObjectName("PageTitle"); self.layout.addWidget(self.heading)
        controls = QHBoxLayout(); self.range_label = QLabel(); self.range = QComboBox()
        self.equipment_label = QLabel(); self.equipment = QComboBox(); self.aggregation_label = QLabel(); self.aggregation = QComboBox()
        self.chart_label = QLabel(); self.chart_filter = QComboBox()
        for control in (self.range, self.equipment, self.aggregation, self.chart_filter): control.currentIndexChanged.connect(self.refresh)
        for widget in (self.range_label, self.range, self.equipment_label, self.equipment, self.aggregation_label, self.aggregation, self.chart_label, self.chart_filter): controls.addWidget(widget)
        controls.addStretch(1); self.layout.addLayout(controls)
        self.metadata = QLabel(); self.metadata.setObjectName("Muted"); self.metadata.setWordWrap(True); self.layout.addWidget(self.metadata)
        grid = QGridLayout(); self.cards = {key: MetricCard(title) for key, title in (("energy", "energy_total"), ("peak", "energy_peak"), ("cop", "analysis_cop"), ("dt", "analysis_delta_t"))}
        for i, card in enumerate(self.cards.values()): grid.addWidget(card, 0, i)
        self.layout.addLayout(grid)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True); self.scroll.setFrameShape(QFrame.NoFrame)
        self.content = QWidget(); self.chart_layout = QVBoxLayout(self.content); self.chart_layout.setSpacing(SPACING_MD); self.chart_layout.addStretch(1); self.scroll.setWidget(self.content); self.layout.addWidget(self.scroll, 1)
        self.retranslate_ui()

    def retranslate_ui(self):
        self.heading.setText(tr("energy_analysis")); current_range = self.range.currentData() or "all"; current_aggregation = self.aggregation.currentData() or "auto"; current_chart = self.chart_filter.currentData() or "all"
        for combo, entries, selected in (
            (self.range, [("all", tr("energy_range_all")), ("24h", tr("energy_range_24h")), ("7d", tr("energy_range_7d")), ("30d", tr("energy_range_30d"))], current_range),
            (self.aggregation, [("auto", tr("energy_aggregation_auto")), ("hour", tr("energy_aggregation_hour")), ("day", tr("energy_aggregation_day")), ("week", tr("energy_aggregation_week")), ("month", tr("energy_aggregation_month"))], current_aggregation),
            (self.chart_filter, [("all", tr("energy_chart_all")), *[(code, tr(title)) for code, title, _ in self.CHARTS]], current_chart),
        ):
            combo.blockSignals(True); combo.clear()
            for value, label in entries: combo.addItem(label, value)
            combo.setCurrentIndex(max(0, combo.findData(selected))); combo.blockSignals(False)
        self.range_label.setText(tr("energy_time_range")); self.equipment_label.setText(tr("analysis_equipment_filter")); self.aggregation_label.setText(tr("energy_aggregation")); self.chart_label.setText(tr("energy_metric"))
        for card in self.cards.values(): card.retranslate_ui()
        self.refresh()

    def _result(self):
        if self.context.dataframe is None or not self.context.semantic_result or not self.context.current_project: return None
        frame = self.context.dataframe; time_name = self.context.import_metadata.get("time_column"); option = self.range.currentData(); selected = self.equipment.currentData()
        if time_name in frame.columns and option != "all":
            timestamps = pd.to_datetime(frame[time_name], errors="coerce"); hours = {"24h": 24, "7d": 168, "30d": 720}[option]
            frame = frame.loc[timestamps >= timestamps.max() - pd.Timedelta(hours=hours)].copy()
        analytics = self.context.diagnosis_result.analytics if self.context.diagnosis_result else None
        return self.context.energy_analysis.analyze(frame, self.context.semantic_result, self.context.current_project.project_id, self.context.import_metadata, self.context.equipment_organization, analytics, selected, self.aggregation.currentData())

    def refresh(self):
        if not hasattr(self, "chart_layout"): return
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
        selected_chart = self.chart_filter.currentData()
        for code, title_key, kind in self.CHARTS:
            if selected_chart != "all" and code != selected_chart: continue
            payload = result.charts.get(code)
            if not payload: continue
            card = QFrame(); card.setObjectName("Card"); box = QVBoxLayout(card); title = QLabel(tr(title_key)); title.setObjectName("CardTitle"); box.addWidget(title)
            unit = QLabel(payload.get("unit", "")); unit.setObjectName("Muted"); box.addWidget(unit)
            chart = TimeSeriesChart(kind); chart.set_payload(payload); box.addWidget(chart); self.chart_layout.insertWidget(self.chart_layout.count()-1, card)
        if result.warnings or q["warnings"]:
            notice = QLabel(tr("energy_quality_warning", warnings=", ".join([*q["warnings"], *result.warnings][:5]))); notice.setObjectName("Muted"); notice.setWordWrap(True); self.chart_layout.insertWidget(0, notice)
