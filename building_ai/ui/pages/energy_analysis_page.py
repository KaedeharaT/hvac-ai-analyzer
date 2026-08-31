"""Interactive, data-driven Energy Analysis page.

This page contains presentation only.  All calculations and capability
decisions remain in :mod:`building_ai.services.energy_analysis_service`.
"""
from __future__ import annotations

import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFontMetrics, QPainter, QPen
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
    """Small, dependency-free engineering chart renderer.

    The renderer consumes measured service payloads only.  It deliberately
    exposes the chart axes, units, scope and series names rather than relying
    on a coloured line to communicate engineering meaning.
    """
    def __init__(self, kind: str = "line"):
        super().__init__(); self.kind = kind; self.payload = {}; self.setMinimumHeight(235)

    def set_payload(self, payload):
        self.payload = payload or {}; self.update()

    @staticmethod
    def _number(value: float, unit: str = "") -> str:
        text = f"{value:,.1f}" if abs(value) >= 100 else f"{value:.2f}" if abs(value) < 10 else f"{value:.1f}"
        return f"{text} {unit}".strip()

    def _axes(self, painter: QPainter):
        outer = self.rect().adjusted(0, 2, -4, -2)
        # A generous left gutter keeps engineering units and numeric ticks from
        # colliding at compact desktop widths; the right gutter is reserved for
        # the heatmap colourbar.
        plot = outer.adjusted(94, 28, -78, -38)
        painter.setPen(QPen(QColor("#E2E8F0"), 1))
        for part in range(1, 5):
            y = plot.top() + part * plot.height() // 5
            painter.drawLine(plot.left(), y, plot.right(), y)
        painter.setPen(QPen(QColor("#94A3B8"), 1))
        painter.drawLine(plot.left(), plot.bottom(), plot.right(), plot.bottom())
        painter.drawLine(plot.left(), plot.top(), plot.left(), plot.bottom())
        painter.setPen(QPen(QColor("#475569")))
        painter.drawText(plot.left(), outer.bottom() - 20, plot.width(), 18, Qt.AlignCenter, self.payload.get("x_label", tr("energy_axis_time")))
        painter.save(); painter.translate(18, plot.center().y()); painter.rotate(-90)
        painter.drawText(-plot.height() // 2, -10, plot.height(), 20, Qt.AlignCenter, self.payload.get("y_label", self.payload.get("unit", "")))
        painter.restore()
        return plot

    def _tick_values(self, painter: QPainter, plot, low: float, high: float, unit: str = "") -> None:
        span = high - low or 1.0
        painter.setPen(QPen(QColor("#64748B")))
        for part in range(0, 6):
            value = low + span * part / 5
            y = plot.bottom() - part * plot.height() // 5
            painter.drawText(27, y - 8, 52, 16, Qt.AlignRight | Qt.AlignVCenter, self._number(value))

    @staticmethod
    def _time_label(value: str) -> str:
        try:
            stamp = pd.Timestamp(value)
            return stamp.strftime("%H:%M") if stamp.hour or stamp.minute else stamp.strftime("%m-%d")
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _scope_time(value: str | None) -> str:
        try:
            return pd.Timestamp(value).strftime("%Y-%m-%d")
        except (TypeError, ValueError):
            return value or "—"

    @staticmethod
    def _legend_label(value: str) -> str:
        translations = {
            "Supply water temperature": "energy_legend_supply_water",
            "Return water temperature": "energy_legend_return_water",
            "Supply air temperature": "energy_legend_supply_air",
            "Return air temperature": "energy_legend_return_air",
            "Outdoor air temperature": "energy_legend_outdoor_air",
            "Input power": "energy_legend_input_power",
            "Energy consumption": "energy_legend_energy",
            "Building total energy": "energy_legend_total_energy",
            "Building total power": "energy_legend_total_power",
            "Weekday": "energy_legend_weekday", "Weekend": "energy_legend_weekend",
        }
        for source, key in translations.items():
            if value == source: return tr(key)
            if value.endswith(": " + source): return value[:-len(source)] + tr(key)
        return value

    def _draw_line(self, painter: QPainter, plot) -> None:
        series = self.payload.get("series", []); palette = ("#2563EB", "#D97706", "#16A34A", "#9333EA")
        values = [point["value"] for item in series for point in item.get("data", []) if point.get("value") is not None]
        if not values:
            painter.drawText(plot, Qt.AlignCenter, tr("energy_no_data")); return
        low, high = min(values), max(values); padding = (high - low) * .08 or max(abs(high) * .08, 1.0); low -= padding; high += padding
        self._tick_values(painter, plot, low, high, self.payload.get("unit", ""))
        for si, item in enumerate(series):
            points = item.get("data", []); painter.setPen(QPen(QColor(palette[si % len(palette)]), 2)); previous = None
            for ix, point in enumerate(points):
                if point.get("value") is None: previous = None; continue
                x = plot.left() + int(ix * plot.width() / max(1, len(points) - 1)); y = plot.bottom() - int((point["value"] - low) / (high - low) * plot.height())
                if previous: painter.drawLine(previous[0], previous[1], x, y)
                previous = (x, y)
            legend_x = plot.left() + (si % 3) * max(120, plot.width() // 3); legend_y = 4 + (si // 3) * 16
            painter.setPen(QPen(QColor(palette[si % len(palette)]), 3)); painter.drawLine(legend_x, legend_y + 7, legend_x + 14, legend_y + 7)
            painter.setPen(QPen(QColor("#334155"))); width = max(84, plot.width() // 3 - 22)
            label = QFontMetrics(painter.font()).elidedText(self._legend_label(item.get("name", "")), Qt.ElideRight, width)
            painter.drawText(legend_x + 19, legend_y, width, 15, Qt.AlignLeft, label)
        first = next((item.get("data", []) for item in series if item.get("data")), [])
        if first:
            for part in (0, 0.5, 1):
                index = min(len(first) - 1, int((len(first) - 1) * part)); x = plot.left() + int(plot.width() * part)
                painter.setPen(QPen(QColor("#64748B"))); painter.drawText(x - 34, plot.bottom() + 3, 68, 16, Qt.AlignHCenter, self._time_label(first[index].get("time", "")))

    def _draw_bars(self, painter: QPainter, plot) -> None:
        data = self.payload.get("data", []); values = [item.get("value", 0.) for item in data]
        if not values:
            painter.drawText(plot, Qt.AlignCenter, tr("energy_no_data")); return
        low, high = 0., max(values) * 1.12 or 1.; self._tick_values(painter, plot, low, high, self.payload.get("unit", ""))
        width = max(8, int(plot.width() / max(1, len(data)) * .62))
        for index, item in enumerate(data):
            center = plot.left() + int((index + .5) * plot.width() / len(data)); height = int(item["value"] / high * plot.height())
            painter.fillRect(center - width // 2, plot.bottom() - height, width, height, QColor("#2563EB" if index == 0 else "#0F766E"))
            painter.setPen(QPen(QColor("#334155"))); painter.drawText(center - 44, plot.bottom() + 3, 88, 16, Qt.AlignHCenter, item["name"])
            painter.drawText(center - 44, plot.bottom() - height - 18, 88, 16, Qt.AlignHCenter, self._number(item["value"], self.payload.get("unit", "")))

    def _draw_scatter(self, painter: QPainter, plot) -> None:
        data = self.payload.get("data", []); xs = [x["x"] for x in data]; ys = [x["y"] for x in data]
        if len(xs) < 3 or max(xs) == min(xs) or max(ys) == min(ys): painter.drawText(plot, Qt.AlignCenter, tr("energy_no_data")); return
        xlow, xhigh = min(xs), max(xs); ylow, yhigh = min(ys), max(ys); self._tick_values(painter, plot, ylow, yhigh, self.payload.get("y_unit", ""))
        painter.setPen(QPen(QColor("#64748B"))); painter.drawText(plot.left(), plot.bottom() + 3, 70, 16, Qt.AlignLeft, self._number(xlow, self.payload.get("x_unit", ""))); painter.drawText(plot.right() - 70, plot.bottom() + 3, 70, 16, Qt.AlignRight, self._number(xhigh, self.payload.get("x_unit", "")))
        painter.setPen(QPen(QColor("#2563EB"), 3))
        for x, y in zip(xs, ys): painter.drawPoint(plot.left() + int((x - xlow) / (xhigh - xlow) * plot.width()), plot.bottom() - int((y - ylow) / (yhigh - ylow) * plot.height()))

    def _draw_heatmap(self, painter: QPainter, plot) -> None:
        data = self.payload.get("data", []); values = [x["value"] for x in data]; maximum = max(values) if values else 0
        dates = list(dict.fromkeys(x["date"] for x in data)); times = list(dict.fromkeys(x["time"] for x in data))
        if not maximum or not dates or not times: painter.drawText(plot, Qt.AlignCenter, tr("energy_no_data")); return
        lookup = {(x["date"], x["time"]): x["value"] for x in data}; width = max(1, plot.width() // len(times)); height = max(1, plot.height() // len(dates))
        for iy, day in enumerate(dates):
            for ix, clock in enumerate(times):
                value = lookup.get((day, clock)); ratio = 0 if value is None else value / maximum
                painter.fillRect(plot.left() + ix * width, plot.top() + iy * height, width, height, QColor(int(238 - 140 * ratio), int(246 - 105 * ratio), int(255 - 24 * ratio)))
        painter.setPen(QPen(QColor("#64748B")))
        for ix in range(0, len(times), max(1, len(times) // 6)):
            painter.drawText(plot.left() + ix * width - 28, plot.bottom() + 3, 56, 16, Qt.AlignHCenter, times[ix])
        for iy in range(0, len(dates), max(1, len(dates) // 4)):
            painter.drawText(1, plot.top() + iy * height - 7, 48, 16, Qt.AlignRight, str(dates[iy])[-5:])
        color_x = plot.right() + 12
        for index in range(plot.height()):
            ratio = 1 - index / max(1, plot.height() - 1); painter.setPen(QPen(QColor(int(238 - 140 * ratio), int(246 - 105 * ratio), int(255 - 24 * ratio)))); painter.drawLine(color_x, plot.top() + index, color_x + 10, plot.top() + index)
        painter.setPen(QPen(QColor("#64748B"))); painter.drawText(color_x + 13, plot.top(), 36, 16, Qt.AlignLeft, self._number(maximum, self.payload.get("unit", ""))); painter.drawText(color_x + 13, plot.bottom() - 14, 36, 16, Qt.AlignLeft, self._number(0, self.payload.get("unit", "")))
        colorbar_label = self.payload.get("colorbar_label", self.payload.get("unit", ""))
        if colorbar_label:
            painter.save(); painter.translate(color_x + 56, plot.center().y()); painter.rotate(-90)
            painter.drawText(-plot.height() // 2, -8, plot.height(), 16, Qt.AlignCenter, colorbar_label)
            painter.restore()

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self); painter.setRenderHint(QPainter.Antialiasing)
        if not self.payload or not (self.payload.get("series") or self.payload.get("data")):
            painter.setPen(QPen(QColor("#64748B"))); painter.drawText(self.rect(), Qt.AlignCenter, tr("energy_no_data")); return
        plot = self._axes(painter)
        if self.kind == "bar": self._draw_bars(painter, plot)
        elif self.kind == "scatter": self._draw_scatter(painter, plot)
        elif self.kind == "heatmap": self._draw_heatmap(painter, plot)
        else: self._draw_line(painter, plot)


class EnergyAnalysisPage(QWidget):
    """Interactive real-data energy view with scoped range/device/aggregation."""
    CHART_CAPABILITIES = {
        "energy_trend": "energy_consumption", "power_trend": "power_trend",
        "temperature_trend": "temperature_trend", "delta_t_trend": "delta_t",
        "cop_trend": "cop", "daily_load_profile": "daily_profile",
        "load_heatmap": "heatmap", "weather_correlation": "weather_correlation",
        "equipment_ranking": "equipment_ranking", "period_comparison": "period_comparison",
    }
    CHARTS = (
        ("energy_trend", "energy_chart_energy", "line", "energy_axis_time", "energy_axis_interval_energy"),
        ("power_trend", "energy_chart_power", "line", "energy_axis_time", "energy_axis_power"),
        ("temperature_trend", "energy_chart_temperature", "line", "energy_axis_time", "energy_axis_temperature"),
        ("delta_t_trend", "energy_chart_delta_t", "line", "energy_axis_time", "energy_axis_delta_t"),
        ("cop_trend", "energy_chart_cop", "line", "energy_axis_time", "energy_axis_cop"),
        ("daily_load_profile", "energy_chart_profile", "line", "energy_axis_hour", "energy_axis_average_power"),
        ("load_heatmap", "energy_chart_heatmap", "heatmap", "energy_axis_hour", "energy_axis_date"),
        ("weather_correlation", "energy_chart_weather", "scatter", "energy_axis_outdoor_temperature", "energy_axis_building_power"),
        ("equipment_ranking", "energy_chart_ranking", "bar", "energy_axis_equipment", "energy_axis_energy"),
        ("period_comparison", "energy_chart_period", "bar", "energy_axis_period", "energy_axis_average_power"),
    )

    @staticmethod
    def _capability_reason(reason: str) -> str:
        requirements = {
            "timestamp_unavailable": "energy_reason_timestamp",
            "requires_energy_meter_or_supported_power_unit": "energy_reason_energy",
            "requires_supported_power_timeseries": "energy_reason_power",
            "requires_mapped_temperature_timeseries": "energy_reason_temperature",
            "requires_supply_return_flow_and_power": "energy_reason_cop",
            "requires_outdoor_temperature_and_power": "energy_reason_weather",
            "requires_two_or_more_equipment_energy_series": "energy_reason_ranking",
            "duplicate_timestamps_require_resolution": "energy_reason_duplicate_time",
        }
        return tr(requirements.get(reason, "energy_no_data"))

    @staticmethod
    def _temperature_axis_label(payload: dict) -> str:
        units = {str(item.get("unit") or "").strip() for item in payload.get("series", [])}
        units.discard("")
        if len(units) == 1:
            unit = next(iter(units))
            return tr("energy_axis_temperature_with_unit", unit=unit)
        return tr("energy_axis_temperature_mixed")

    @staticmethod
    def _quality_warning_text(warning: str) -> str:
        code = str(warning).split(":", 1)[0]
        keys = {
            "unknown_or_unsupported_energy_unit": "energy_warning_energy_unit",
            "unknown_or_unsupported_power_unit": "energy_warning_power_unit",
            "negative_energy": "energy_warning_negative_energy",
            "negative_power": "energy_warning_negative_power",
            "impossible_temperature": "energy_warning_temperature_range",
            "outlier_values": "energy_warning_outliers",
        }
        return tr(keys.get(code, "energy_warning_generic"))

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
        self.content = QWidget()
        # Charts are intentionally arranged as a responsive desktop grid.  The
        # service still owns all data and capability decisions; this page only
        # decides how an available result is presented.
        self.chart_grid = QGridLayout(self.content)
        self.chart_grid.setContentsMargins(0, 0, 0, 0)
        self.chart_grid.setHorizontalSpacing(SPACING_MD)
        self.chart_grid.setVerticalSpacing(SPACING_MD)
        self.chart_grid.setColumnStretch(0, 1); self.chart_grid.setColumnStretch(1, 1)
        # Compatibility alias for integrations that used the older layout name.
        self.chart_layout = self.chart_grid
        self.scroll.setWidget(self.content); self.layout.addWidget(self.scroll, 1)
        self.retranslate_ui()

    def retranslate_ui(self):
        self.heading.setText(tr("energy_analysis")); current_range = self.range.currentData() or "all"; current_aggregation = self.aggregation.currentData() or "auto"; current_chart = self.chart_filter.currentData() or "all"
        for combo, entries, selected in (
            (self.range, [("all", tr("energy_range_all")), ("24h", tr("energy_range_24h")), ("7d", tr("energy_range_7d")), ("30d", tr("energy_range_30d"))], current_range),
            (self.aggregation, [("auto", tr("energy_aggregation_auto")), ("hour", tr("energy_aggregation_hour")), ("day", tr("energy_aggregation_day")), ("week", tr("energy_aggregation_week")), ("month", tr("energy_aggregation_month"))], current_aggregation),
            (self.chart_filter, [("all", tr("energy_chart_all")), *[(code, tr(title)) for code, title, _, _, _ in self.CHARTS]], current_chart),
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
        while self.chart_grid.count():
            child = self.chart_grid.takeAt(0)
            if child.widget(): child.widget().deleteLater()
        if result is None:
            self.metadata.setText(tr("energy_no_data")); return
        q = result.data_quality; self.metadata.setText(tr("energy_metadata", start=result.start or "—", end=result.end or "—", interval=(f"{result.sampling_interval_minutes:g}" if result.sampling_interval_minutes else "—"), equipment=result.summary["equipment_count"], energy=result.summary["energy_points"], temperature=result.summary["temperature_points"], missing=q["missing_ratio"]))
        summary = result.summary
        self.cards["energy"].set_value("—" if summary["total_energy_kwh"] is None else f"{summary['total_energy_kwh']:,.1f} kWh")
        self.cards["peak"].set_value("—" if summary["peak_power_kw"] is None else f"{summary['peak_power_kw']:,.1f} kW")
        self.cards["cop"].set_value("—" if summary["average_cop"] is None else f"{summary['average_cop']:.2f}")
        self.cards["dt"].set_value("—" if summary["average_delta_t_c"] is None else f"{summary['average_delta_t_c']:.2f} °C")
        selected_chart = self.chart_filter.currentData()
        row, column = 0, 0
        for code, title_key, kind, x_key, y_key in self.CHARTS:
            if selected_chart != "all" and code != selected_chart: continue
            payload = result.charts.get(code)
            if not payload:
                if selected_chart != "all":
                    card = QFrame(); card.setObjectName("Card"); box = QVBoxLayout(card)
                    title = QLabel(tr(title_key)); title.setObjectName("CardTitle"); box.addWidget(title)
                    detail = result.capability_details.get(self.CHART_CAPABILITIES.get(code, ""), {})
                    message = QLabel(tr("energy_chart_unavailable", reason=self._capability_reason(detail.get("reason", "")))); message.setObjectName("Muted"); message.setWordWrap(True); box.addWidget(message)
                    self.chart_grid.addWidget(card, row, column, 1, 2)
                continue
            payload = dict(payload)
            payload["x_label"] = tr(x_key); payload["y_label"] = tr(y_key)
            if code == "temperature_trend":
                payload["y_label"] = self._temperature_axis_label(payload)
            elif code == "weather_correlation":
                payload["x_label"] = tr("energy_axis_outdoor_temperature_with_unit", unit=payload.get("x_unit") or "—")
                payload["y_label"] = tr("energy_axis_building_power_with_unit", unit=payload.get("y_unit") or "—")
            elif code == "load_heatmap":
                payload["colorbar_label"] = tr("energy_axis_power")
            card = QFrame(); card.setObjectName("Card"); box = QVBoxLayout(card); title = QLabel(tr(title_key)); title.setObjectName("CardTitle"); box.addWidget(title)
            scope = QLabel(tr("energy_chart_scope", start=TimeSeriesChart._scope_time(result.start), end=TimeSeriesChart._scope_time(result.end), interval=(f"{result.sampling_interval_minutes:g}" if result.sampling_interval_minutes else "—"), equipment=(self.equipment.currentText() if self.equipment.currentData() else tr("analysis_all_equipment")))); scope.setObjectName("Muted"); scope.setWordWrap(True); box.addWidget(scope)
            chart = TimeSeriesChart(kind); chart.set_payload(payload); box.addWidget(chart)
            # Trends and heatmaps benefit from the full reading width; paired
            # comparison charts remain side-by-side on ordinary desktop sizes.
            full_width = code in {"energy_trend", "load_heatmap"}
            if full_width and column:
                row += 1; column = 0
            span = 2 if full_width else 1
            self.chart_grid.addWidget(card, row, column, 1, span)
            if full_width:
                row += 1
            else:
                column += 1
                if column == 2:
                    row += 1; column = 0
        if result.warnings or q["warnings"]:
            warnings = list(dict.fromkeys([*q["warnings"], *result.warnings]))[:5]
            detail = "; ".join(dict.fromkeys(self._quality_warning_text(item) for item in warnings))
            notice = QLabel(tr("energy_quality_warning", warnings=detail)); notice.setObjectName("Muted"); notice.setWordWrap(True)
            # Place quality guidance below the data visuals so it is never
            # mistaken for a chart or an engineering conclusion.
            if column:
                row += 1
            self.chart_grid.addWidget(notice, row, 0, 1, 2)
