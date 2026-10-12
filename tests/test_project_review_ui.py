import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt5.QtWidgets import QApplication
import pandas as pd

from building_ai.config import Settings
from building_ai.ui.context import ApplicationContext
from building_ai.ui.pages.pages import SemanticsPage
from building_ai.i18n import LanguageManager


def test_review_fields_do_not_show_another_project_and_language_preserves_edits(tmp_path, monkeypatch):
    monkeypatch.setenv('BUILDING_AI_CONFIG_DIR', str(tmp_path / 'config'))
    app = QApplication.instance() or QApplication([])
    context = ApplicationContext(Settings(data_dir=tmp_path / 'data'))
    first = context.projects.create('A'); context.open_project(first.project_id)
    path = tmp_path / 'test.csv'
    pd.DataFrame({'time': ['2025-01-01'], 'CH-1 Power kW': [10]}).to_csv(path, index=False)
    context.import_data(str(path)); context.run_semantics()
    page = SemanticsPage(context); page.show(); app.processEvents(); page.table.selectRow(0); app.processEvents()
    assert page.equipment_id.text() == 'CH-1', (page.table.currentRow(), page.table.rowCount(), page.table.signalsBlocked(), context.semantic_result.semantic_results)
    page.note.setText('unsaved QA note')
    LanguageManager.instance().set_language('ja_JP')
    assert page.note.text() == 'unsaved QA note'
    second = context.projects.create('B'); context.open_project(second.project_id)
    page.refresh()
    assert not page.equipment_id.text() and not page.unit.text() and not page.note.text()
    assert not page.details.toPlainText() and page.table.currentRow() == -1
    page.close(); app.processEvents()
