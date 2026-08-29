"""Central Qt stylesheet; page code uses semantic object names instead of inline styles."""

from .theme import COLORS, CONTROL_HEIGHT, FONT_BODY, FONT_HERO_METRIC, FONT_METRIC, FONT_PAGE_TITLE, FONT_SECTION_TITLE, FONT_SMALL, FONT_TITLE, RADIUS_LG, RADIUS_MD, RADIUS_SM


def application_stylesheet() -> str:
    c = COLORS
    return f"""
    QWidget {{ background: {c['background']}; color: {c['text']}; font-family: 'Segoe UI', 'Microsoft YaHei UI', 'Yu Gothic UI', sans-serif; font-size: {FONT_BODY}px; }}
    QMainWindow {{ background: {c['background']}; }}
    QLabel#AppTitle {{ color: white; font-size: {FONT_TITLE}px; font-weight: 700; background: transparent; }}
    QLabel#PageTitle {{ font-size: {FONT_PAGE_TITLE}px; font-weight: 700; letter-spacing: -0.2px; background: transparent; }}
    QLabel#SectionTitle {{ font-size: {FONT_SECTION_TITLE}px; font-weight: 650; background: transparent; }}
    QLabel#Muted, QLabel#StatusLabel {{ color: {c['muted']}; background: transparent; }}
    QFrame#Sidebar {{ background: {c['sidebar']}; border: 0; }}
    QLabel#SidebarSection {{ color: #7E8BA0; font-size: 10px; font-weight: 700; letter-spacing: 1px; background: transparent; padding: 11px 12px 3px 12px; }}
    QLabel#SidebarSubtitle {{ color: #8FA0B8; font-size: 11px; background: transparent; padding: 0 12px; }}
    QFrame#Topbar {{ background: {c['surface']}; border-bottom: 1px solid {c['border']}; }}
    QFrame#Card {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: {RADIUS_LG}px; }}
    QFrame#ChartCard {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: {RADIUS_LG}px; }}
    QFrame#EmptyState {{ background: {c['surface']}; border: 1px dashed #CDD6E2; border-radius: {RADIUS_LG}px; min-height: 160px; }}
    QLabel#EmptyStateTitle {{ font-size: {FONT_SECTION_TITLE}px; font-weight: 650; background: transparent; }}
    QFrame#ChatTranscript {{ background: {c['background']}; border: 0; }}
    QFrame#ChatMessageAssistant {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: {RADIUS_MD}px; }}
    QFrame#ChatMessageUser {{ background: {c['selected']}; border: 1px solid #C9D9F8; border-radius: {RADIUS_MD}px; }}
    QLabel#ChatMessageRole {{ color: {c['muted']}; font-size: {FONT_SMALL}px; font-weight: 600; background: transparent; }}
    QLabel#ChatMessageBody {{ background: transparent; }}
    QFrame#ToolCall {{ background: #F8FAFC; border: 1px solid {c['border']}; border-radius: {RADIUS_SM}px; }}
    QFrame#AgentSetupNotice {{ background: #FFFBEB; border: 1px solid #FDE68A; border-radius: {RADIUS_MD}px; }}
    QFrame#AgentSuggestions {{ background: #F8FAFC; border: 1px solid {c['border']}; border-radius: {RADIUS_MD}px; }}
    QLabel#CardValue {{ font-size: {FONT_METRIC}px; font-weight: 700; background: transparent; }}
    QLabel#CardTitle {{ color: {c['muted']}; background: transparent; }}
    QLabel#StatusBadge {{ border-radius: 12px; padding: 2px 9px; font-size: 11px; font-weight: 650; }}
    QLabel#StatusBadge[tone="success"] {{ background: #EAF7EF; color: {c['success']}; }}
    QLabel#StatusBadge[tone="warning"] {{ background: #FFF6E8; color: {c['warning']}; }}
    QLabel#StatusBadge[tone="critical"] {{ background: #FDECEC; color: {c['critical']}; }}
    QLabel#StatusBadge[tone="info"] {{ background: #EAF1FF; color: {c['info']}; }}
    QLabel#StatusBadge[tone="neutral"] {{ background: #F1F5F9; color: {c['muted']}; }}
    QPushButton {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: {RADIUS_SM}px; padding: 0 12px; min-height: {CONTROL_HEIGHT}px; font-weight: 550; }}
    QPushButton:hover {{ background: {c['hover']}; }}
    QPushButton#PrimaryButton {{ background: {c['primary']}; color: white; border: 1px solid {c['primary']}; font-weight: 600; }}
    QPushButton#PrimaryButton:hover {{ background: {c['primary_hover']}; }}
    QPushButton#DangerButton {{ color: {c['danger']}; border-color: #FED7AA; background: #FFF7ED; }}
    QPushButton#TextButton {{ background: transparent; color: {c['primary']}; border: 0; padding: 3px 0; }}
    QPushButton#TextButton:hover {{ background: transparent; color: {c['primary_hover']}; text-decoration: underline; }}
    QPushButton#SuggestionButton {{ background: {c['surface']}; text-align: left; color: {c['text']}; }}
    QTextEdit#ChatInput {{ padding: 9px; }}
    QPushButton#NavButton {{ color: #B9C5D6; background: transparent; border: 0; border-radius: {RADIUS_SM}px; padding: 0 12px; min-height: 38px; text-align: left; font-weight: 550; }}
    QPushButton#NavButton:hover {{ background: {c['sidebar_hover']}; color: white; }}
    QPushButton#NavButton:checked {{ background: #213455; color: white; }}
    QLineEdit, QComboBox, QTextEdit, QTableWidget, QTreeWidget, QListWidget {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: {RADIUS_SM}px; padding: 6px 8px; selection-background-color: {c['primary']}; selection-color: white; }}
    QLineEdit, QComboBox {{ min-height: {CONTROL_HEIGHT - 8}px; }}
    QTableWidget, QTreeWidget {{ gridline-color: {c['border']}; alternate-background-color: #FAFCFF; }}
    QHeaderView::section {{ background: #F8FAFC; color: #475569; border: 0; border-bottom: 1px solid {c['border']}; padding: 10px 8px; font-weight: 650; }}
    QTableWidget::item {{ padding: 7px; border-bottom: 1px solid #F0F3F7; }} QTableWidget::item:hover {{ background: {c['hover']}; }}
    QTabWidget::pane {{ border: 0; }}
    QTabBar::tab {{ background: transparent; color: {c['muted']}; padding: 8px 14px; border: 0; border-bottom: 2px solid transparent; }}
    QTabBar::tab:selected {{ color: {c['primary']}; border-bottom-color: {c['primary']}; font-weight: 600; }}
    QProgressBar {{ background: #E2E8F0; border: 0; border-radius: 5px; text-align: center; color: {c['text']}; font-size: 10px; }}
    QProgressBar::chunk {{ background: {c['primary']}; border-radius: 5px; }}
    QStatusBar {{ background: {c['surface']}; border-top: 1px solid {c['border']}; color: {c['muted']}; }}
    """
