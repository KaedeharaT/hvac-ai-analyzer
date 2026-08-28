"""Central Qt stylesheet; page code uses semantic object names instead of inline styles."""

from .theme import COLORS, FONT_BODY, FONT_SMALL, FONT_TITLE


def application_stylesheet() -> str:
    c = COLORS
    return f"""
    QWidget {{ background: {c['background']}; color: {c['text']}; font-family: 'Segoe UI', 'Microsoft YaHei'; font-size: {FONT_BODY}px; }}
    QMainWindow {{ background: {c['background']}; }}
    QLabel#AppTitle {{ color: white; font-size: {FONT_TITLE}px; font-weight: 700; background: transparent; }}
    QLabel#PageTitle {{ font-size: 26px; font-weight: 700; background: transparent; }}
    QLabel#Muted, QLabel#StatusLabel {{ color: {c['muted']}; background: transparent; }}
    QFrame#Sidebar {{ background: {c['sidebar']}; border: 0; }}
    QFrame#Topbar {{ background: {c['surface']}; border-bottom: 1px solid {c['border']}; }}
    QFrame#Card {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 10px; }}
    QFrame#ChatTranscript {{ background: {c['background']}; border: 0; }}
    QFrame#ChatMessageAssistant {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 8px; }}
    QFrame#ChatMessageUser {{ background: {c['selected']}; border: 1px solid #C9D9F8; border-radius: 8px; }}
    QLabel#ChatMessageRole {{ color: {c['muted']}; font-size: {FONT_SMALL}px; font-weight: 600; background: transparent; }}
    QLabel#ChatMessageBody {{ background: transparent; }}
    QFrame#ToolCall {{ background: #F8FAFC; border: 1px solid {c['border']}; border-radius: 6px; }}
    QFrame#AgentSetupNotice {{ background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 8px; }}
    QFrame#AgentSuggestions {{ background: #F8FAFC; border: 1px solid {c['border']}; border-radius: 8px; }}
    QLabel#CardValue {{ font-size: 24px; font-weight: 700; background: transparent; }}
    QLabel#CardTitle {{ color: {c['muted']}; background: transparent; }}
    QPushButton {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 6px; padding: 7px 12px; }}
    QPushButton:hover {{ background: {c['hover']}; }}
    QPushButton#PrimaryButton {{ background: {c['primary']}; color: white; border: 1px solid {c['primary']}; font-weight: 600; }}
    QPushButton#PrimaryButton:hover {{ background: {c['primary_hover']}; }}
    QPushButton#DangerButton {{ color: {c['danger']}; border-color: #FECACA; }}
    QPushButton#TextButton {{ background: transparent; color: {c['primary']}; border: 0; padding: 3px 0; }}
    QPushButton#TextButton:hover {{ background: transparent; color: {c['primary_hover']}; text-decoration: underline; }}
    QPushButton#SuggestionButton {{ background: {c['surface']}; text-align: left; color: {c['text']}; }}
    QTextEdit#ChatInput {{ padding: 9px; }}
    QPushButton#NavButton {{ color: #CBD5E1; background: transparent; border: 0; border-radius: 6px; padding: 10px 12px; text-align: left; font-weight: 600; }}
    QPushButton#NavButton:hover {{ background: #1F2937; color: white; }}
    QPushButton#NavButton:checked {{ background: #263B63; color: white; }}
    QLineEdit, QComboBox, QTextEdit, QTableWidget, QTreeWidget, QListWidget {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 6px; padding: 6px; selection-background-color: {c['primary']}; selection-color: white; }}
    QTableWidget, QTreeWidget {{ gridline-color: {c['border']}; alternate-background-color: #FAFCFF; }}
    QHeaderView::section {{ background: #F1F5F9; color: #475569; border: 0; border-bottom: 1px solid {c['border']}; padding: 8px; font-weight: 600; }}
    QTableWidget::item {{ padding: 5px; }} QTableWidget::item:hover {{ background: {c['hover']}; }}
    QTabWidget::pane {{ border: 0; }}
    QTabBar::tab {{ background: transparent; color: {c['muted']}; padding: 8px 14px; border: 0; border-bottom: 2px solid transparent; }}
    QTabBar::tab:selected {{ color: {c['primary']}; border-bottom-color: {c['primary']}; font-weight: 600; }}
    QProgressBar {{ background: #E2E8F0; border: 0; border-radius: 5px; text-align: center; color: {c['text']}; font-size: 10px; }}
    QProgressBar::chunk {{ background: {c['primary']}; border-radius: 5px; }}
    QStatusBar {{ background: {c['surface']}; border-top: 1px solid {c['border']}; color: {c['muted']}; }}
    """
