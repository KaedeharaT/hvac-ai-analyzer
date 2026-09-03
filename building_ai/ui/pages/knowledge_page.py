"""Calm, read-only entry point for BuildingAI's curated knowledge corpus."""
from __future__ import annotations

import json

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy, QTableWidget,
    QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from building_ai.i18n import LanguageManager, tr
from building_ai.knowledge import KnowledgeService
from building_ai.knowledge_catalog import CATALOG_DIR, source_registry
from building_ai.ui.components import SectionHeader, StatusBadge
from building_ai.ui.theme import SPACING_LG, SPACING_MD, SPACING_SM, SPACING_XS


class _MetricCard(QFrame):
    """A concise metric card that grows safely with DPI-scaled typography."""
    def __init__(self, key: str):
        super().__init__(); self.setObjectName("KnowledgeMetricCard"); self.key = key
        self.setMinimumHeight(116); self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout = QVBoxLayout(self); layout.setContentsMargins(SPACING_MD, 13, SPACING_MD, 13); layout.setSpacing(5)
        self.title = QLabel(); self.title.setObjectName("CardTitle"); self.title.setWordWrap(True)
        self.value = QLabel("—"); self.value.setObjectName("KnowledgeMetricValue")
        layout.addWidget(self.title); layout.addStretch(1); layout.addWidget(self.value); self.retranslate_ui()
    def set_value(self, value: str) -> None: self.value.setText(value)
    def retranslate_ui(self) -> None: self.title.setText(tr(self.key))


class _FeaturedSourceCard(QFrame):
    def __init__(self, source: dict, purpose_key: str, open_callback):
        super().__init__(); self.setObjectName("FeaturedSourceCard"); self.source = source; self.purpose_key = purpose_key
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout = QVBoxLayout(self); layout.setContentsMargins(13, 12, 13, 12); layout.setSpacing(4)
        self.organization = QLabel(source.get("organization", "")); self.organization.setObjectName("FeaturedSourceOrganization")
        self.title = QLabel(source.get("title", "")); self.title.setObjectName("FeaturedSourceTitle"); self.title.setWordWrap(True)
        self.purpose = QLabel(); self.purpose.setObjectName("Muted"); self.purpose.setWordWrap(True)
        self.badge = QLabel(); self.badge.setObjectName("KnowledgeMeta")
        self.open_button = QPushButton(); self.open_button.setObjectName("TextButton"); self.open_button.clicked.connect(lambda: open_callback(source.get("official_url", "")))
        for widget in (self.organization, self.title, self.purpose, self.badge, self.open_button): layout.addWidget(widget)
        self.retranslate_ui()
    def retranslate_ui(self) -> None:
        self.purpose.setText(tr(self.purpose_key)); categories = self.source.get("knowledge_category", [])
        category = next((item for item in ("semantic", "system_topology", "operation_maintenance", "energy_saving", "retrofit", "energy_analysis") if item in categories), "engineering_principles")
        self.badge.setText(f"{tr('knowledge_country_' + self.source.get('country', 'Global'))} · {tr('knowledge_category_' + category)}")
        self.open_button.setText(tr("knowledge_view_source"))


class KnowledgeBasePage(QWidget):
    """Continuous knowledge dashboard; retrieval delegates to KnowledgeService."""
    COUNTRY_OPTIONS = ("all", "China", "US", "Japan", "Global")
    LANGUAGE_OPTIONS = ("all", "Chinese", "English", "Japanese", "Multilingual")
    CATEGORY_OPTIONS = ("all", "semantic", "energy_saving", "engineering_principles", "operation_maintenance", "retrofit", "energy_analysis", "operation", "zeb", "control", "equipment", "system_topology")
    FEATURED = (("us_energyplus", "knowledge_feature_energyplus"), ("us_doe_femp_om", "knowledge_feature_femp"), ("us_project_haystack", "knowledge_feature_haystack"), ("us_brick", "knowledge_feature_brick"), ("jp_env_zeb_retrofit", "knowledge_feature_japan"), ("cn_mee_public_institutions", "knowledge_feature_china"))

    def __init__(self, context):
        super().__init__(); self.context = context; self.i18n = LanguageManager.instance(); self._sources = source_registry(); self._source_by_id = {item["source_id"]: item for item in self._sources}
        self._manifest = self._load_manifest(); self._results: list[dict] = []; self._featured_cards: list[_FeaturedSourceCard] = []
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True); self.scroll.setFrameShape(QFrame.NoFrame); outer.addWidget(self.scroll)
        self.content = QWidget(); self.content.setObjectName("KnowledgePage"); self.scroll.setWidget(self.content)
        self.layout = QVBoxLayout(self.content); self.layout.setContentsMargins(SPACING_LG, SPACING_LG, SPACING_LG, SPACING_LG); self.layout.setSpacing(SPACING_LG)
        # Search and topics are the primary user tasks. Corpus statistics are
        # deliberately secondary product metadata rather than the hero.
        self._build_search_hero(); self._build_results(); self._build_knowledge_sections(); self._build_featured_sources(); self._build_metrics(); self._build_source_browser(); self.layout.addStretch(1)
        self.i18n.language_changed.connect(self.retranslate_ui); self.retranslate_ui(); self.refresh()

    @staticmethod
    def _load_manifest() -> dict:
        try: return json.loads((CATALOG_DIR / "metadata" / "catalog_manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): return {"source_count": 0, "chunk_count": 0, "chunks_by_country": {}, "chunks_by_category": {}}

    def _build_search_hero(self) -> None:
        hero = QFrame(); hero.setObjectName("KnowledgeSearchHero"); layout = QVBoxLayout(hero); layout.setContentsMargins(SPACING_LG, SPACING_MD, SPACING_LG, SPACING_MD); layout.setSpacing(SPACING_SM)
        top = QHBoxLayout(); self.search_heading = QLabel(); self.search_heading.setObjectName("SectionTitle"); self.status = StatusBadge(); top.addWidget(self.search_heading); top.addStretch(1); top.addWidget(self.status); layout.addLayout(top)
        controls = QHBoxLayout(); controls.setSpacing(SPACING_SM); self.query = QLineEdit(); self.query.setMinimumHeight(44); self.query.returnPressed.connect(self.search); self.search_button = QPushButton(); self.search_button.setObjectName("PrimaryButton"); self.search_button.setMinimumHeight(44); self.search_button.clicked.connect(self.search); controls.addWidget(self.query, 1); controls.addWidget(self.search_button); layout.addLayout(controls)
        self.search_support = QLabel(); self.search_support.setObjectName("Muted"); layout.addWidget(self.search_support); self.layout.addWidget(hero)

    def _build_metrics(self) -> None:
        self.metric_grid = QGridLayout(); self.metric_grid.setHorizontalSpacing(SPACING_MD); self.metric_grid.setVerticalSpacing(SPACING_MD)
        self.cards = {"knowledge_chunks": _MetricCard("knowledge_chunks"), "knowledge_trusted_sources": _MetricCard("knowledge_trusted_sources"), "knowledge_regions": _MetricCard("knowledge_regions"), "knowledge_languages": _MetricCard("knowledge_languages")}
        self.layout.addLayout(self.metric_grid); self._reflow_metrics()

    def _build_results(self) -> None:
        self.results_section = QFrame(); self.results_section.setObjectName("KnowledgeResultsSection"); layout = QVBoxLayout(self.results_section); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(SPACING_SM)
        self.result_status = QLabel(); self.result_status.setObjectName("SectionTitle"); layout.addWidget(self.result_status)
        filters = QHBoxLayout(); filters.setSpacing(SPACING_SM); self.country_filter = QComboBox(); self.category_filter = QComboBox(); self.language_filter = QComboBox()
        for combo, options, prefix in ((self.country_filter, self.COUNTRY_OPTIONS, "knowledge_country_"), (self.category_filter, self.CATEGORY_OPTIONS, "knowledge_category_"), (self.language_filter, self.LANGUAGE_OPTIONS, "knowledge_language_")):
            combo.setMinimumHeight(34); combo.setProperty("translation_prefix", prefix)
            for value in options: combo.addItem("", value)
            combo.currentIndexChanged.connect(self._render_results); filters.addWidget(combo)
        filters.addStretch(1); layout.addLayout(filters)
        self.results_holder = QWidget(); self.results_layout = QVBoxLayout(self.results_holder); self.results_layout.setContentsMargins(0, SPACING_XS, 0, 0); self.results_layout.setSpacing(SPACING_SM); layout.addWidget(self.results_holder); self.layout.addWidget(self.results_section); self.results_section.setVisible(False)

    def _build_knowledge_sections(self) -> None:
        areas = QFrame(); areas.setObjectName("KnowledgeSection"); areas_layout = QVBoxLayout(areas); areas_layout.setContentsMargins(SPACING_MD, SPACING_MD, SPACING_MD, SPACING_MD); areas_layout.setSpacing(SPACING_SM); self.areas_header = SectionHeader("", ""); areas_layout.addWidget(self.areas_header)
        self.category_rows = {}
        for category in self.CATEGORY_OPTIONS[1:]:
            line = QWidget(); line_layout = QHBoxLayout(line); line_layout.setContentsMargins(0, 1, 0, 1); line_layout.setSpacing(SPACING_SM); name = QLabel(); name.setMinimumWidth(130); name.setWordWrap(True); bar = QProgressBar(); bar.setTextVisible(False); bar.setFixedHeight(7); bar.setMaximum(1); value = QLabel("0"); value.setObjectName("Muted"); value.setMinimumWidth(24); value.setAlignment(Qt.AlignRight | Qt.AlignVCenter); line_layout.addWidget(name); line_layout.addWidget(bar, 1); line_layout.addWidget(value); areas_layout.addWidget(line); self.category_rows[category] = (name, bar, value)
        self.layout.addWidget(areas)
        coverage = QFrame(); coverage.setObjectName("KnowledgeSection"); coverage_layout = QVBoxLayout(coverage); coverage_layout.setContentsMargins(SPACING_MD, SPACING_MD, SPACING_MD, SPACING_MD); coverage_layout.setSpacing(SPACING_SM); self.coverage_header = SectionHeader("", ""); coverage_layout.addWidget(self.coverage_header)
        coverage_grid = QGridLayout(); coverage_grid.setHorizontalSpacing(SPACING_LG); coverage_grid.setVerticalSpacing(SPACING_SM)
        self.region_rows = {}
        for index, country in enumerate(("China", "US", "Japan", "Global")):
            line = QWidget(); line_layout = QHBoxLayout(line); line_layout.setContentsMargins(0, 3, 0, 3); line_layout.setSpacing(SPACING_SM); name = QLabel(); name.setMinimumWidth(78); bar = QProgressBar(); bar.setTextVisible(False); bar.setFixedHeight(7); bar.setMaximum(1); value = QLabel("0"); value.setObjectName("Muted"); value.setMinimumWidth(70); value.setAlignment(Qt.AlignRight | Qt.AlignVCenter); line_layout.addWidget(name); line_layout.addWidget(bar, 1); line_layout.addWidget(value); self.region_rows[country] = (name, bar, value)
            coverage_grid.addWidget(line, index // 2, index % 2)
        coverage_layout.addLayout(coverage_grid); self.layout.addWidget(coverage)

    def _build_featured_sources(self) -> None:
        header = QHBoxLayout(); self.featured_header = SectionHeader("", ""); self.browse_sources_button = QPushButton(); self.browse_sources_button.setObjectName("TextButton"); self.browse_sources_button.clicked.connect(self._toggle_source_browser); header.addWidget(self.featured_header, 1); header.addWidget(self.browse_sources_button); self.layout.addLayout(header)
        self.featured_grid = QGridLayout(); self.featured_grid.setHorizontalSpacing(SPACING_MD); self.featured_grid.setVerticalSpacing(SPACING_MD)
        for source_id, purpose_key in self.FEATURED:
            if source := self._source_by_id.get(source_id): self._featured_cards.append(_FeaturedSourceCard(source, purpose_key, self._open_url))
        self.layout.addLayout(self.featured_grid); self._reflow_featured()

    def _build_source_browser(self) -> None:
        self.source_browser = QFrame(); self.source_browser.setObjectName("KnowledgeSourceBrowser"); layout = QVBoxLayout(self.source_browser); layout.setContentsMargins(SPACING_MD, SPACING_MD, SPACING_MD, SPACING_MD); layout.setSpacing(SPACING_SM); self.source_browser_header = SectionHeader("", ""); layout.addWidget(self.source_browser_header)
        self.source_table = QTableWidget(0, 5); self.source_table.setObjectName("KnowledgeSourceTable"); self.source_table.setEditTriggers(QTableWidget.NoEditTriggers); self.source_table.setSelectionBehavior(QTableWidget.SelectRows); self.source_table.setAlternatingRowColors(True); self.source_table.verticalHeader().setDefaultSectionSize(32); self.source_table.horizontalHeader().setStretchLastSection(True); self.source_table.setMinimumHeight(240); self.source_table.itemSelectionChanged.connect(self._show_selected_source); layout.addWidget(self.source_table)
        details = QHBoxLayout(); self.source_details = QTextEdit(); self.source_details.setReadOnly(True); self.source_details.setMinimumHeight(112); self.source_url_button = QPushButton(); self.source_url_button.setObjectName("TextButton"); self.source_url_button.clicked.connect(self._open_selected_source); details.addWidget(self.source_details, 1); details.addWidget(self.source_url_button, alignment=Qt.AlignBottom); layout.addLayout(details); self.layout.addWidget(self.source_browser); self.source_browser.setVisible(False)

    def refresh(self) -> None:
        stats = KnowledgeService(self.context.database).stats(); sources, chunks = int(self._manifest.get("source_count", 0)), int(self._manifest.get("chunk_count", 0)); ready = stats.get("chunk_count", 0) >= chunks > 0; self.status.set_status(tr("knowledge_ready") if ready else tr("knowledge_not_ready"), "success" if ready else "warning")
        self.cards["knowledge_chunks"].set_value(str(chunks)); self.cards["knowledge_trusted_sources"].set_value(str(sources)); self.cards["knowledge_regions"].set_value("3"); self.cards["knowledge_languages"].set_value("3")
        category_counts = self._manifest.get("chunks_by_category", {}); max_category = max(category_counts.values(), default=1)
        for category, (name, bar, value) in self.category_rows.items():
            count = int(category_counts.get(category, 0)); name.setText(tr("knowledge_category_" + category)); bar.setMaximum(max_category); bar.setValue(count); value.setText(str(count))
        country_counts = self._manifest.get("chunks_by_country", {}); max_country = max(country_counts.values(), default=1)
        for country, (name, bar, value) in self.region_rows.items():
            count = int(country_counts.get(country, 0)); name.setText(tr("knowledge_country_" + country)); bar.setMaximum(max_country); bar.setValue(count); value.setText(tr("knowledge_source_chunk_count", chunks=count))
        self._populate_sources()

    def search(self) -> None:
        query = self.query.text().strip(); self._results = KnowledgeService(self.context.database).search(query, top_k=100) if query else []; self.results_section.setVisible(bool(query)); self._render_results()
        if query: self.scroll.ensureWidgetVisible(self.results_section, 0, SPACING_MD)

    def _filtered_results(self) -> list[dict]:
        country, category, language = self.country_filter.currentData(), self.category_filter.currentData(), self.language_filter.currentData()
        return [result for result in self._results if (country == "all" or result["metadata"].get("country") == country) and (category == "all" or result["metadata"].get("knowledge_category") == category) and (language == "all" or result["metadata"].get("language") == language)]

    def _render_results(self) -> None:
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        results = self._filtered_results(); self.result_status.setText(tr("knowledge_search_results", count=len(results)))
        for result in results: self.results_layout.addWidget(self._result_card(result))

    def _result_card(self, result: dict) -> QFrame:
        metadata = result.get("metadata", {}); card = QFrame(); card.setObjectName("KnowledgeResultCard"); card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum); layout = QVBoxLayout(card); layout.setContentsMargins(14, 12, 14, 12); layout.setSpacing(4)
        org = QLabel(metadata.get("organization", "")); org.setObjectName("FeaturedSourceOrganization"); title = QLabel(result.get("title", "")); title.setObjectName("FeaturedSourceTitle"); title.setWordWrap(True); meta = QLabel(f"{tr('knowledge_category_' + metadata.get('knowledge_category', 'semantic'))} · {tr('knowledge_country_' + metadata.get('country', 'Global'))} · {tr('knowledge_language_' + metadata.get('language', 'Multilingual'))}"); meta.setObjectName("KnowledgeMeta"); excerpt = QLabel(result.get("text", "")); excerpt.setObjectName("Muted"); excerpt.setWordWrap(True); excerpt.setTextInteractionFlags(Qt.TextSelectableByMouse); button = QPushButton(tr("knowledge_view_source")); button.setObjectName("TextButton"); button.clicked.connect(lambda: self._open_url(metadata.get("official_url") or result.get("source", "")))
        for widget in (org, title, meta, excerpt, button): layout.addWidget(widget)
        return card

    @staticmethod
    def _source_category_text(source: dict) -> str:
        categories = source.get("knowledge_category", []); categories = [categories] if isinstance(categories, str) else categories; display = ("semantic", "system_topology", "operation_maintenance", "energy_saving", "retrofit", "energy_analysis", "operation", "zeb", "control", "equipment"); selected = [category for category in display if category in categories]
        return ", ".join(tr("knowledge_category_" + category) for category in selected[:2]) or tr("knowledge_category_engineering_principles")

    def _populate_sources(self) -> None:
        self.source_table.setRowCount(len(self._sources))
        for row, source in enumerate(self._sources):
            values = (source.get("organization", ""), source.get("title", ""), tr("knowledge_country_" + source.get("country", "Global")), self._source_category_text(source), source.get("language", ""))
            for column, value in enumerate(values): self.source_table.setItem(row, column, QTableWidgetItem(value))
        if self._sources and not self.source_table.selectedItems(): self.source_table.selectRow(0)
        if self._sources: self._show_selected_source()

    def _show_selected_source(self) -> None:
        row = self.source_table.currentRow()
        if not 0 <= row < len(self._sources): return
        source = self._sources[row]; chunk_count = sum(item.get("source_id") == source.get("source_id") for item in self._chunk_rows())
        self.source_details.setPlainText("\n".join((f"{source.get('organization', '')} — {source.get('title', '')}", f"{tr('knowledge_source_purpose')}: {self._source_category_text(source)}", f"{tr('knowledge_source_language')}: {source.get('language', '')} · {tr('knowledge_source_chunks')}: {chunk_count}", f"{tr('knowledge_source_usage')}: {source.get('license_or_usage_note', '')}", f"{tr('knowledge_source_strategy')}: {source.get('content_strategy', '')}", f"URL: {source.get('official_url', '')}")))

    @staticmethod
    def _chunk_rows() -> list[dict]:
        try: return [json.loads(line) for line in (CATALOG_DIR / "chunks" / "knowledge_chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        except OSError: return []

    def _open_selected_source(self) -> None:
        row = self.source_table.currentRow()
        if 0 <= row < len(self._sources): self._open_url(self._sources[row].get("official_url", ""))
    def _toggle_source_browser(self) -> None:
        self.source_browser.setVisible(not self.source_browser.isVisible()); self.browse_sources_button.setText(tr("knowledge_hide_sources") if self.source_browser.isVisible() else tr("knowledge_browse_sources"))
        if self.source_browser.isVisible(): self.scroll.ensureWidgetVisible(self.source_browser, 0, SPACING_MD)
    @staticmethod
    def _open_url(url: str) -> None:
        if isinstance(url, str) and url.startswith(("https://", "http://")): QDesktopServices.openUrl(QUrl(url))
    @staticmethod
    def _clear_grid(grid: QGridLayout) -> None:
        while grid.count(): grid.takeAt(0)
    def _reflow_metrics(self) -> None:
        if not hasattr(self, "metric_grid"): return
        self._clear_grid(self.metric_grid); columns = 2 if self.width() < 940 else 4
        for index, card in enumerate(self.cards.values()): self.metric_grid.addWidget(card, index // columns, index % columns)
        for column in range(columns): self.metric_grid.setColumnStretch(column, 1)
    def _reflow_featured(self) -> None:
        if not hasattr(self, "featured_grid"): return
        self._clear_grid(self.featured_grid); columns = 1 if self.width() < 740 else 2 if self.width() < 1180 else 3
        for index, card in enumerate(self._featured_cards): self.featured_grid.addWidget(card, index // columns, index % columns)
        for column in range(columns): self.featured_grid.setColumnStretch(column, 1)
    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event); self._reflow_metrics(); self._reflow_featured()
    def retranslate_ui(self) -> None:
        self.search_heading.setText(tr("knowledge_search_heading")); self.query.setPlaceholderText(tr("knowledge_search_placeholder")); self.search_button.setText(tr("knowledge_search")); self.search_support.setText(tr("knowledge_search_support")); self.areas_header.title.setText(tr("knowledge_areas")); self.areas_header.subtitle.setText(tr("knowledge_areas_subtitle")); self.coverage_header.title.setText(tr("knowledge_coverage")); self.coverage_header.subtitle.setText(tr("knowledge_coverage_subtitle")); self.featured_header.title.setText(tr("knowledge_featured_sources")); self.featured_header.subtitle.setText(tr("knowledge_featured_subtitle")); self.browse_sources_button.setText(tr("knowledge_hide_sources") if self.source_browser.isVisible() else tr("knowledge_browse_sources")); self.source_browser_header.title.setText(tr("knowledge_sources")); self.source_browser_header.subtitle.setText(tr("knowledge_sources_intro")); self.source_url_button.setText(tr("knowledge_open_official")); self.source_table.setHorizontalHeaderLabels([tr("knowledge_organization"), tr("knowledge_source"), tr("knowledge_country"), tr("knowledge_category"), tr("knowledge_language")])
        for card in self.cards.values(): card.retranslate_ui()
        for card in self._featured_cards: card.retranslate_ui()
        for combo in (self.country_filter, self.category_filter, self.language_filter):
            prefix = combo.property("translation_prefix")
            for index in range(combo.count()): combo.setItemText(index, tr(prefix + combo.itemData(index)))
        self.refresh(); self._render_results()
