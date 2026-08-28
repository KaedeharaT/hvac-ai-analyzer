from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QButtonGroup, QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QStackedWidget, QVBoxLayout, QWidget,
)

from building_ai.i18n import LanguageManager, tr

from .context import ApplicationContext
from .pages import (
    AgentPage, AnalyticsPage, DashboardPage, DataPage, EnergyAnalysisPage, EquipmentPage,
    ProjectsPage, SemanticsPage, SettingsPage,
)
from .styles import application_stylesheet
from .theme import SPACING_LG, SPACING_MD, SPACING_SM


class MainWindow(QMainWindow):
    """Application shell with a persistent status-aware header and sidebar."""

    NAVIGATION = (
        ("▦", "dashboard", DashboardPage), ("▣", "projects", ProjectsPage),
        ("⇧", "import_data", DataPage), ("◇", "semantic_mapping", SemanticsPage),
        ("⌘", "equipment", EquipmentPage), ("▤", "energy_analysis", EnergyAnalysisPage), ("▤", "analysis", AnalyticsPage),
        ("✦", "agent", AgentPage), ("⚙", "settings", SettingsPage),
    )

    def __init__(self, context: ApplicationContext | None = None):
        super().__init__()
        self.context = context or ApplicationContext()
        self.i18n = LanguageManager.instance()
        self.i18n.set_language(self.context.settings.language)
        self.i18n.language_changed.connect(self.retranslate_ui)
        self.setMinimumSize(1040, 680)
        self.resize(1280, 820)
        self.setStyleSheet(application_stylesheet())

        root = QWidget()
        shell = QHBoxLayout(root); shell.setContentsMargins(0, 0, 0, 0); shell.setSpacing(0)
        shell.addWidget(self._build_sidebar())
        content = QWidget(); content_layout = QVBoxLayout(content); content_layout.setContentsMargins(0, 0, 0, 0); content_layout.setSpacing(0)
        content_layout.addWidget(self._build_topbar())
        self.stack = QStackedWidget()
        self.pages = [page(self.context) for _, _, page in self.NAVIGATION]
        for page in self.pages:
            self.stack.addWidget(page)
        content_layout.addWidget(self.stack, 1)
        shell.addWidget(content, 1)
        self.setCentralWidget(root)

        self.pages[1].project_changed.connect(self.refresh_pages)
        self.pages[2].data_changed.connect(self.refresh_pages)
        self.navigation_buttons[0].setChecked(True)
        self.change_page(0)
        self.statusBar().showMessage(tr("read_only_notice"))
        self.retranslate_ui()

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame(); sidebar.setObjectName("Sidebar"); sidebar.setFixedWidth(212)
        layout = QVBoxLayout(sidebar); layout.setContentsMargins(SPACING_MD, SPACING_LG, SPACING_MD, SPACING_MD); layout.setSpacing(SPACING_SM)
        self.app_title = QLabel(); self.app_title.setObjectName("AppTitle"); layout.addWidget(self.app_title)
        self.app_subtitle = QLabel("Building Energy Intelligence"); self.app_subtitle.setObjectName("Muted"); self.app_subtitle.setStyleSheet("color: #94A3B8;")
        layout.addWidget(self.app_subtitle); layout.addSpacing(SPACING_MD)
        self.nav_group = QButtonGroup(self); self.nav_group.setExclusive(True); self.navigation_buttons: list[QPushButton] = []
        for index, (icon, key, _) in enumerate(self.NAVIGATION):
            button = QPushButton(); button.setObjectName("NavButton"); button.setCheckable(True); button.setMinimumHeight(40)
            button.clicked.connect(lambda checked=False, i=index: self.change_page(i))
            button.setProperty("translation_key", key); button.setProperty("nav_icon", icon)
            self.nav_group.addButton(button); self.navigation_buttons.append(button); layout.addWidget(button)
        layout.addStretch(1)
        self.sidebar_hint = QLabel("Local-first • Read-only"); self.sidebar_hint.setObjectName("Muted"); self.sidebar_hint.setStyleSheet("color: #94A3B8;")
        layout.addWidget(self.sidebar_hint)
        return sidebar

    def _build_topbar(self) -> QFrame:
        bar = QFrame(); bar.setObjectName("Topbar"); layout = QHBoxLayout(bar); layout.setContentsMargins(SPACING_LG, SPACING_SM, SPACING_LG, SPACING_SM)
        self.page_title = QLabel(); self.page_title.setObjectName("PageTitle"); layout.addWidget(self.page_title)
        layout.addStretch(1)
        self.project_label = QLabel(); self.project_label.setObjectName("StatusLabel"); layout.addWidget(self.project_label)
        self.llm_label = QLabel(); self.llm_label.setObjectName("StatusLabel"); layout.addWidget(self.llm_label)
        self.language_button = QPushButton(); self.language_button.clicked.connect(self.toggle_language); layout.addWidget(self.language_button)
        return bar

    def toggle_language(self) -> None:
        target = "zh_CN" if self.i18n.language == "en_US" else "en_US"
        self.i18n.set_language(target)
        self.context.settings.language = target
        self.context.settings.save()

    def change_page(self, index: int) -> None:
        if not 0 <= index < len(self.pages):
            return
        self.stack.setCurrentIndex(index)
        self.navigation_buttons[index].setChecked(True)
        self.page_title.setText(tr(self.NAVIGATION[index][1]))
        page = self.pages[index]
        if hasattr(page, "refresh"):
            page.refresh()
        self.update_status()

    def update_status(self) -> None:
        project = self.context.current_project
        self.project_label.setText(f"{tr('current_project')}: {project.name if project else tr('no_project')}")
        provider = self.context.llm_manager.get_provider()
        if not provider.is_configured:
            detail = tr("not_configured")
            color = "#D97706"
        else:
            ok, _ = provider.test_connection(timeout=0.35)
            detail = tr("connected") if ok else tr("connection_error")
            color = "#16A34A" if ok else "#DC2626"
        self.llm_label.setText(f"{tr('llm')}: {provider.display_name}  <span style='color:{color}'>●</span> {detail}")

    def refresh_pages(self) -> None:
        for page in self.pages:
            if hasattr(page, "refresh"):
                page.refresh()
        self.update_status()

    def retranslate_ui(self) -> None:
        self.setWindowTitle(tr("app_name"))
        self.app_title.setText(tr("app_name"))
        self.language_button.setText("中文" if self.i18n.language == "en_US" else "EN")
        for button in self.navigation_buttons:
            button.setText(f"{button.property('nav_icon')}  {tr(button.property('translation_key'))}")
        index = self.stack.currentIndex() if hasattr(self, "stack") else 0
        self.page_title.setText(tr(self.NAVIGATION[index][1]))
        for page in getattr(self, "pages", []):
            if hasattr(page, "retranslate_ui"):
                page.retranslate_ui()
        self.statusBar().showMessage(tr("read_only_notice"))
        self.update_status()
