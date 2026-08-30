"""User-facing browser for BuildingAI's versioned local knowledge corpus."""
from __future__ import annotations

import json
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import (
    QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QScrollArea, QTabWidget, QTableWidget, QTableWidgetItem,
    QTextEdit, QVBoxLayout, QWidget,
)

from building_ai.i18n import LanguageManager, tr
from building_ai.knowledge import KnowledgeService
from building_ai.knowledge_catalog import CATALOG_DIR, source_registry
from building_ai.ui.components import StatusBadge
from building_ai.ui.theme import SPACING_LG, SPACING_MD, SPACING_SM


class _MetricCard(QFrame):
    def __init__(self, key: str):
        super().__init__(); self.setObjectName("Card"); self.key = key
        layout = QVBoxLayout(self); layout.setContentsMargins(SPACING_MD, SPACING_MD, SPACING_MD, SPACING_MD); layout.setSpacing(4)
        self.title = QLabel(); self.title.setObjectName("CardTitle")
        self.value = QLabel("—"); self.value.setObjectName("CardValue")
        self.note = QLabel(); self.note.setObjectName("Muted"); self.note.setWordWrap(True)
        layout.addWidget(self.title); layout.addWidget(self.value); layout.addWidget(self.note)
        self.retranslate_ui()

    def set_value(self, value: str, note: str = "") -> None:
        self.value.setText(value); self.note.setText(note)

    def retranslate_ui(self) -> None:
        self.title.setText(tr(self.key))


class KnowledgeBasePage(QWidget):
    """Lightweight read-only discovery UI; retrieval always uses KnowledgeService."""

    COUNTRY_OPTIONS = ("all", "China", "US", "Japan", "Global")
    LANGUAGE_OPTIONS = ("all", "Chinese", "English", "Japanese", "Multilingual")
    CATEGORY_OPTIONS = (
        "all", "semantic", "energy_saving", "engineering_principles",
        "operation_maintenance", "retrofit", "energy_analysis", "operation",
        "zeb", "control", "equipment", "system_topology",
    )

    def __init__(self, context):
        super().__init__()
        self.context = context; self.i18n = LanguageManager.instance()
        self._sources = source_registry()
        self._manifest = self._load_manifest()
        self._results: list[dict] = []
        self.layout = QVBoxLayout(self); self.layout.setContentsMargins(SPACING_LG, SPACING_LG, SPACING_LG, SPACING_LG); self.layout.setSpacing(SPACING_MD)
        self.heading = QLabel(); self.heading.setObjectName("PageTitle"); self.layout.addWidget(self.heading)
        self.description = QLabel(); self.description.setObjectName("Muted"); self.description.setWordWrap(True); self.layout.addWidget(self.description)
        header = QHBoxLayout(); self.status = StatusBadge(); header.addWidget(self.status); header.addStretch(1); self.layout.addLayout(header)
        self.tabs = QTabWidget(); self.layout.addWidget(self.tabs, 1)
        self._build_overview(); self._build_search(); self._build_sources()
        self.i18n.language_changed.connect(self.retranslate_ui)
        self.retranslate_ui(); self.refresh()

    @staticmethod
    def _load_manifest() -> dict:
        path = CATALOG_DIR / "metadata" / "catalog_manifest.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"source_count": 0, "chunk_count": 0, "chunks_by_country": {}, "chunks_by_language": {}, "chunks_by_category": {}}

    def _build_overview(self) -> None:
        page = QWidget(); layout = QVBoxLayout(page); layout.setSpacing(SPACING_MD)
        grid = QGridLayout(); grid.setHorizontalSpacing(SPACING_MD); grid.setVerticalSpacing(SPACING_MD)
        self.cards = {key: _MetricCard(key) for key in ("knowledge_sources", "knowledge_chunks", "knowledge_regions", "knowledge_languages")}
        for index, card in enumerate(self.cards.values()): grid.addWidget(card, index // 2, index % 2)
        layout.addLayout(grid)
        self.regions_title = QLabel(); self.regions_title.setObjectName("SectionTitle"); layout.addWidget(self.regions_title)
        self.region_grid = QGridLayout(); self.region_grid.setHorizontalSpacing(SPACING_MD); self.region_grid.setVerticalSpacing(SPACING_MD)
        self.region_cards = {country: _MetricCard(f"knowledge_country_{country}") for country in ("China", "US", "Japan", "Global")}
        for index, card in enumerate(self.region_cards.values()): self.region_grid.addWidget(card, index // 2, index % 2)
        layout.addLayout(self.region_grid)
        self.categories_title = QLabel(); self.categories_title.setObjectName("SectionTitle"); layout.addWidget(self.categories_title)
        self.category_grid = QGridLayout(); self.category_grid.setHorizontalSpacing(SPACING_SM); self.category_grid.setVerticalSpacing(SPACING_SM)
        self.category_labels: dict[str, QLabel] = {}
        for index, category in enumerate(self.CATEGORY_OPTIONS[1:]):
            label = QLabel(); label.setObjectName("KnowledgeCategoryBadge"); self.category_labels[category] = label
            self.category_grid.addWidget(label, index // 3, index % 3)
        layout.addLayout(self.category_grid); layout.addStretch(1); self.tabs.addTab(page, "")

    def _build_search(self) -> None:
        page = QWidget(); layout = QVBoxLayout(page); layout.setSpacing(SPACING_MD)
        controls = QHBoxLayout(); self.query = QLineEdit(); self.query.returnPressed.connect(self.search)
        self.search_button = QPushButton(); self.search_button.setObjectName("PrimaryButton"); self.search_button.clicked.connect(self.search)
        controls.addWidget(self.query, 1); controls.addWidget(self.search_button); layout.addLayout(controls)
        filters = QHBoxLayout(); self.country_filter = QComboBox(); self.category_filter = QComboBox(); self.language_filter = QComboBox()
        for combo, options, prefix in ((self.country_filter, self.COUNTRY_OPTIONS, "knowledge_country_"), (self.category_filter, self.CATEGORY_OPTIONS, "knowledge_category_"), (self.language_filter, self.LANGUAGE_OPTIONS, "knowledge_language_")):
            for value in options: combo.addItem("", value)
            combo.currentIndexChanged.connect(self._render_results); filters.addWidget(combo)
            combo.setProperty("translation_prefix", prefix)
        filters.addStretch(1); layout.addLayout(filters)
        self.result_status = QLabel(); self.result_status.setObjectName("Muted"); layout.addWidget(self.result_status)
        self.result_holder = QWidget(); self.result_layout = QVBoxLayout(self.result_holder); self.result_layout.setContentsMargins(0, 0, 0, 0); self.result_layout.setSpacing(SPACING_SM); self.result_layout.addStretch(1)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame); scroll.setWidget(self.result_holder); layout.addWidget(scroll, 1)
        self.tabs.addTab(page, "")

    def _build_sources(self) -> None:
        page = QWidget(); layout = QVBoxLayout(page); layout.setSpacing(SPACING_MD)
        self.sources_intro = QLabel(); self.sources_intro.setObjectName("Muted"); self.sources_intro.setWordWrap(True); layout.addWidget(self.sources_intro)
        self.source_table = QTableWidget(0, 4); self.source_table.setObjectName("KnowledgeSourceTable")
        self.source_table.setEditTriggers(QTableWidget.NoEditTriggers); self.source_table.setSelectionBehavior(QTableWidget.SelectRows); self.source_table.setAlternatingRowColors(True); self.source_table.verticalHeader().setDefaultSectionSize(34); self.source_table.horizontalHeader().setStretchLastSection(True)
        self.source_table.itemSelectionChanged.connect(self._show_selected_source); layout.addWidget(self.source_table, 1)
        self.source_details = QTextEdit(); self.source_details.setReadOnly(True); self.source_details.setMinimumHeight(125); layout.addWidget(self.source_details)
        self.source_url_button = QPushButton(); self.source_url_button.setObjectName("TextButton"); self.source_url_button.clicked.connect(self._open_selected_source); layout.addWidget(self.source_url_button)
        self.tabs.addTab(page, "")

    def refresh(self) -> None:
        stats = KnowledgeService(self.context.database).stats()
        sources, chunks = self._manifest.get("source_count", 0), self._manifest.get("chunk_count", 0)
        ready = stats.get("chunk_count", 0) >= chunks > 0
        self.status.set_status(tr("knowledge_ready") if ready else tr("knowledge_not_ready"), "success" if ready else "warning")
        self.cards["knowledge_sources"].set_value(str(sources))
        self.cards["knowledge_chunks"].set_value(str(chunks))
        self.cards["knowledge_regions"].set_value(tr("knowledge_regions_value"))
        self.cards["knowledge_languages"].set_value(tr("knowledge_languages_value"))
        country_sources = {country: sum(item.get("country") == country for item in self._sources) for country in self.region_cards}
        for country, card in self.region_cards.items():
            card.set_value(str(country_sources[country]), tr("knowledge_source_chunk_count", chunks=self._manifest.get("chunks_by_country", {}).get(country, 0)))
        for category, label in self.category_labels.items():
            label.setText(f"{tr('knowledge_category_' + category)}  {self._manifest.get('chunks_by_category', {}).get(category, 0)}")
        self._populate_sources()

    def search(self) -> None:
        query = self.query.text().strip()
        self._results = KnowledgeService(self.context.database).search(query, top_k=100) if query else []
        self._render_results()

    def _filtered_results(self) -> list[dict]:
        country = self.country_filter.currentData(); category = self.category_filter.currentData(); language = self.language_filter.currentData()
        return [result for result in self._results if (country == "all" or result["metadata"].get("country") == country) and (category == "all" or result["metadata"].get("knowledge_category") == category) and (language == "all" or result["metadata"].get("language") == language)]

    def _render_results(self) -> None:
        while self.result_layout.count() > 1:
            item = self.result_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        results = self._filtered_results() if hasattr(self, "country_filter") else []
        self.result_status.setText(tr("knowledge_search_results", count=len(results)) if self._results else tr("knowledge_search_hint"))
        for result in results:
            self.result_layout.insertWidget(self.result_layout.count() - 1, self._result_card(result))

    def _result_card(self, result: dict) -> QFrame:
        metadata = result.get("metadata", {}); card = QFrame(); card.setObjectName("KnowledgeResultCard")
        layout = QVBoxLayout(card); layout.setContentsMargins(SPACING_SM, SPACING_SM, SPACING_SM, SPACING_SM); layout.setSpacing(3)
        title = QLabel(result.get("title", "")); title.setObjectName("SectionTitle"); title.setWordWrap(True); layout.addWidget(title)
        provenance = QLabel(f"{metadata.get('organization', '')} · {tr('knowledge_country_' + metadata.get('country', 'Global'))} · {tr('knowledge_category_' + metadata.get('knowledge_category', 'semantic'))}"); provenance.setObjectName("Muted"); provenance.setWordWrap(True); layout.addWidget(provenance)
        excerpt = QLabel(result.get("text", "")); excerpt.setWordWrap(True); excerpt.setTextInteractionFlags(Qt.TextSelectableByMouse); layout.addWidget(excerpt)
        source_button = QPushButton(tr("knowledge_view_source")); source_button.setObjectName("TextButton"); source_button.clicked.connect(lambda checked=False, item=result: self._open_url(item.get("metadata", {}).get("official_url") or item.get("source", ""))); layout.addWidget(source_button)
        return card

    @staticmethod
    def _source_category_text(source: dict) -> str:
        categories = source.get("knowledge_category", [])
        if isinstance(categories, str): categories = [categories]
        display = ("semantic", "system_topology", "operation_maintenance", "energy_saving", "retrofit", "energy_analysis", "operation", "zeb", "control", "equipment")
        selected = [category for category in display if category in categories]
        return ", ".join(tr("knowledge_category_" + category) for category in selected[:3]) or tr("knowledge_category_engineering_principles")

    def _populate_sources(self) -> None:
        self.source_table.setRowCount(len(self._sources))
        for row, source in enumerate(self._sources):
            values = (tr("knowledge_country_" + source.get("country", "Global")), source.get("organization", ""), source.get("title", ""), self._source_category_text(source))
            for column, value in enumerate(values): self.source_table.setItem(row, column, QTableWidgetItem(value))
        if self._sources and not self.source_table.selectedItems(): self.source_table.selectRow(0)

    def _show_selected_source(self) -> None:
        row = self.source_table.currentRow()
        if not 0 <= row < len(self._sources): return
        source = self._sources[row]
        self.source_details.setPlainText("\n".join((
            f"{tr('knowledge_source_purpose')}: {self._source_category_text(source)}",
            f"{tr('knowledge_source_language')}: {source.get('language', '')}",
            f"{tr('knowledge_source_usage')}: {source.get('license_or_usage_note', '')}",
            f"{tr('knowledge_source_strategy')}: {source.get('content_strategy', '')}",
            f"URL: {source.get('official_url', '')}",
        )))

    def _open_selected_source(self) -> None:
        row = self.source_table.currentRow()
        if 0 <= row < len(self._sources): self._open_url(self._sources[row].get("official_url", ""))

    @staticmethod
    def _open_url(url: str) -> None:
        if url.startswith(("https://", "http://")): QDesktopServices.openUrl(QUrl(url))

    def retranslate_ui(self) -> None:
        self.heading.setText(tr("knowledge_base")); self.description.setText(tr("knowledge_description")); self.regions_title.setText(tr("knowledge_region_distribution")); self.categories_title.setText(tr("knowledge_categories"))
        self.tabs.setTabText(0, tr("knowledge_overview")); self.tabs.setTabText(1, tr("knowledge_search")); self.tabs.setTabText(2, tr("knowledge_sources"))
        self.query.setPlaceholderText(tr("knowledge_search_placeholder")); self.search_button.setText(tr("knowledge_search")); self.sources_intro.setText(tr("knowledge_sources_intro")); self.source_url_button.setText(tr("knowledge_view_source"))
        self.source_table.setHorizontalHeaderLabels([tr("knowledge_country"), tr("knowledge_organization"), tr("knowledge_source"), tr("knowledge_category")])
        for card in getattr(self, "cards", {}).values(): card.retranslate_ui()
        for card in getattr(self, "region_cards", {}).values(): card.retranslate_ui()
        for combo in (getattr(self, "country_filter", None), getattr(self, "category_filter", None), getattr(self, "language_filter", None)):
            if combo is None: continue
            prefix = combo.property("translation_prefix")
            for index in range(combo.count()): combo.setItemText(index, tr(prefix + combo.itemData(index)))
        self.refresh(); self._render_results()
