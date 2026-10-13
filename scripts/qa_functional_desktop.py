"""Native desktop acceptance runner. Private inputs and outputs stay local.

Use a JSON manifest containing ``bems`` entries (name/path) and an optional
``model`` and ``drawings`` list. Imports go through real Qt file dialogs.
Never run this against an existing user database.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--reopen', action='store_true')
    parser.add_argument('--workflows-only', action='store_true')
    parser.add_argument('--ai-only', action='store_true')
    parser.add_argument('--assets-only', action='store_true')
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    from building_ai.config import Settings
    original = Settings.load()
    os.environ['BUILDING_AI_CONFIG_DIR'] = str(workspace / 'config')
    os.environ['QT_QPA_PLATFORM'] = 'windows'
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox, QDialogButtonBox, QScrollArea, QLineEdit
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_DontUseNativeDialogs)
    from building_ai.ui.model_viewer_backend import prepare_model_viewer
    prepare_model_viewer()
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    from building_ai.ui.context import ApplicationContext
    from building_ai.ui.main_window import MainWindow
    settings = Settings.load() if (args.reopen or args.workflows_only or args.ai_only) else Settings(provider=original.provider, model=original.model,
                        ollama_url=original.ollama_url, language='en_US',
                        data_dir=workspace / 'runtime')
    settings.save()
    context = ApplicationContext(settings)
    window = MainWindow(context)
    window.setWindowFlag(Qt.WindowStaysOnTopHint)
    window.show()
    report = {'platform': app.platformName(), 'projects': [], 'dialogs': [], 'exceptions': []}
    def save():
        (workspace / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    def exception(kind, value, tb):
        report['exceptions'].append(''.join(traceback.format_exception(kind, value, tb)))
        save()
        print(report['exceptions'][-1], flush=True)
    sys.excepthook = exception
    def events(ms=100):
        QTest.qWait(ms)
        app.processEvents()
    def click(widget):
        parent = widget.parentWidget()
        while parent:
            if isinstance(parent, QScrollArea):
                parent.ensureWidgetVisible(widget)
                events(10)
            parent = parent.parentWidget()
        QTest.mouseClick(widget, Qt.LeftButton)
        events()
    def navigate(key):
        click(window.navigation_buttons[next(i for i, (_, k, _) in enumerate(window.NAVIGATION) if k == key)])
        assert window.stack.currentWidget() is window.page_by_key[key]
    answers = deque()
    heartbeat = [0]
    def dialog_input():
        heartbeat[0] += 1
        dialog = app.activeModalWidget()
        if isinstance(dialog, QMessageBox):
            report['dialogs'].append(dialog.text())
            button = dialog.button(QMessageBox.Ok) or dialog.button(QMessageBox.Yes) or dialog.button(QMessageBox.No)
            if button:
                QTimer.singleShot(0, lambda button=button: QTest.mouseClick(button, Qt.LeftButton))
        elif answers and isinstance(dialog, (QFileDialog, QInputDialog)):
            answer = answers.popleft()
            if isinstance(dialog, QFileDialog):
                dialog.findChild(QLineEdit, 'fileNameEdit').setText(str(Path(answer).resolve()))
                box = dialog.findChild(QDialogButtonBox)
                button = box.button(QDialogButtonBox.Open) or box.button(QDialogButtonBox.Save)
                QTimer.singleShot(0, lambda button=button: QTest.mouseClick(button, Qt.LeftButton))
            else:
                if dialog.inputMode() == QInputDialog.TextInput:
                    dialog.setTextValue(answer)
                box = dialog.findChild(QDialogButtonBox)
                QTimer.singleShot(0, lambda box=box: QTest.mouseClick(box.button(QDialogButtonBox.Ok), Qt.LeftButton))
    watchdog = QTimer()
    watchdog.timeout.connect(dialog_input)
    watchdog.start(100)
    def capture(name):
        window.grab().save(str(workspace / (name + '.png')))
    if args.assets_only:
        from scripts.qa_functional_workflows import exercise_independent_assets
        if args.reopen:
            report = json.loads((workspace / 'report.json').read_text(encoding='utf-8'))
        exercise_independent_assets(window, context, manifest, workspace, report,
                                    navigate, click, events, answers, args.reopen)
        settings.save(); window.close(); events(); save()
        assert not report['exceptions'], 'Unhandled Qt exception; see report.json'
        return 0
    if args.workflows_only or args.ai_only:
        report = json.loads((workspace / 'report.json').read_text(encoding='utf-8'))
        report['previous_exceptions'] = report.pop('exceptions', [])
        report['exceptions'] = []
        navigate('projects'); page = window.page_by_key['projects']
        row = next(i for i in range(page.list.rowCount()) if page.list.item(i, 0).data(Qt.UserRole) == report['projects'][-1]['project_id'])
        page.list.selectRow(row); click(page.open_button)
    if args.ai_only:
        from scripts.qa_functional_workflows import exercise_ai, capture_languages
        exercise_ai(window, context, manifest, workspace, report, navigate, click, events, save)
        capture_languages(window, context, workspace, report, navigate, events, save)
        settings.save(); window.close(); events(); save()
        assert not report['exceptions'], 'Unhandled Qt exception; see report.json'
        return 0
    if args.reopen:
        previous = json.loads((workspace / 'report.json').read_text(encoding='utf-8'))
        report = previous
        for entry in previous['projects']:
            if not entry.get('rows'): continue
            navigate('projects')
            page = window.page_by_key['projects']
            row = next(i for i in range(page.list.rowCount()) if page.list.item(i, 0).data(Qt.UserRole) == entry['project_id'])
            page.list.selectRow(row); click(page.open_button)
            assert len(context.dataframe) == entry['rows']
            saved = {p.raw_name: p for p in context.semantic_result.semantic_results}
            for expected in entry.get('reviews', []):
                assert saved[expected['column']].confirmed_label == expected['label']
                assert saved[expected['column']].unit == expected.get('unit', saved[expected['column']].unit)
        from scripts.qa_functional_workflows import restore_spatial
        restored = restore_spatial(window, context, workspace, navigate, click, events) if manifest.get('workflows') else {'projects': 'PASS', 'manual_reviews': 'PASS'}
        restored['language'] = settings.language
        if manifest.get('retrofit_review'):
            from scripts.qa_drawing_retrofit import exercise_retrofit
            exercise_retrofit(window, context, manifest, workspace, report, navigate, click, events, answers, save, reopen=True)
        (workspace / 'reopen.json').write_text(json.dumps(restored, indent=2), encoding='utf-8')
        window.close(); events()
        assert not report['exceptions'], 'Unhandled Qt exception; see report.json'
        return 0
    for index, entry in enumerate([] if args.workflows_only else manifest['bems']):
        navigate('projects')
        answers.append(entry['name'])
        click(window.page_by_key['projects'].create_button)
        assert not answers
        result = {'name': entry['name'], 'project_id': context.current_project.project_id,
                  'new_project_clean': context.dataframe is None and context.semantic_result is None}
        report['projects'].append(result)
        navigate('import_data')
        answers.append(entry['path'])
        if entry.get('sheet'):
            answers.append(entry['sheet'])
        started = time.monotonic()
        before_ticks = heartbeat[0]
        click(window.page_by_key['import_data'].import_button)
        deadline = time.monotonic() + 300
        page = window.page_by_key['import_data']
        while getattr(page, '_import_worker', None) is not None and time.monotonic() < deadline:
            events(100)
        assert getattr(page, '_import_worker', None) is None, 'Import worker timed out'
        result['import_seconds'] = round(time.monotonic() - started, 2)
        result['import_event_loop_ticks'] = heartbeat[0] - before_ticks
        assert not answers
        result['rows'] = len(context.dataframe) if context.dataframe is not None else None
        if entry.get('expect_error'):
            assert context.dataframe is None and report['dialogs']
            result['expected_error_handled'] = True
            save(); continue
        result['metadata'] = context.import_metadata
        capture(f'{index}-import')
        save()
        if context.dataframe is None:
            continue
        navigate('semantic_mapping')
        click(window.page_by_key['semantic_mapping'].run_button)
        semantics = window.page_by_key['semantic_mapping']
        for review in entry.get('reviews', []):
            row = next(i for i, point in enumerate(context.semantic_result.semantic_results) if point.raw_name == review['column'])
            semantics.table.selectRow(row); events()
            semantics.label.setCurrentText(review['label'])
            semantics.equipment_id.setText(review['equipment'])
            if 'unit' in review: semantics.unit.setText(review['unit'])
            semantics.note.setText('LOCAL QA ONLY: confirmed against fixture/source header; not research GT.')
            click(semantics.accept)
        result['reviews'] = entry.get('reviews', [])
        if context.semantic_result:
            result['semantics'] = [asdict(item) for item in context.semantic_result.semantic_results]
        result['equipment'] = [asdict(item) for item in context.equipment]
        capture(f'{index}-semantics')
        navigate('equipment')
        capture(f'{index}-equipment')
        navigate('energy_analysis')
        if context.energy_analysis_result:
            result['energy'] = asdict(context.energy_analysis_result)
        capture(f'{index}-energy')
        navigate('analysis')
        analysis = window.page_by_key['analysis']
        click(analysis.run_button)
        deadline = time.monotonic() + 180
        while analysis._analysis_worker is not None and time.monotonic() < deadline:
            events(200)
        assert analysis._analysis_worker is None, 'Analysis timed out'
        result['diagnosis_completed'] = context.diagnosis_result is not None
        if context.diagnosis_result:
            result['diagnosis'] = asdict(context.diagnosis_result)
            assert context.diagnosis_result.consistency_validation.passed
        if entry.get('expect_cop'):
            assert context.diagnosis_result.analytics.available_kpis
        if entry.get('no_cop'):
            assert not context.diagnosis_result.analytics.available_kpis
        navigate('energy_analysis')
        energy = window.page_by_key['energy_analysis']
        result['resolutions'] = []
        for resolution in ('1min', '1h', '1d', '1mo'):
            choice = energy.aggregation.findData(resolution)
            if choice >= 0 and energy.aggregation.model().item(choice).isEnabled():
                energy.aggregation.setCurrentIndex(choice); events()
                result['resolutions'].append(resolution)
        if context.energy_analysis_result.start:
            energy.range.setCurrentIndex(energy.range.findData('custom')); events()
            result['custom_range'] = True
        capture(f'{index}-diagnosis')
        save()
        print(json.dumps({k: v for k, v in result.items() if k in ('name', 'new_project_clean', 'rows', 'import_seconds', 'diagnosis_completed')}), flush=True)
    if manifest.get('workflows'):
        from scripts.qa_functional_workflows import exercise
        exercise(window, context, manifest, workspace, report, navigate, click, events, answers, save)
    if manifest.get('retrofit_review'):
        from scripts.qa_drawing_retrofit import exercise_retrofit
        exercise_retrofit(window, context, manifest, workspace, report, navigate, click, events, answers, save)
    settings.save()
    window.close()
    events()
    save()
    assert not report['exceptions'], 'Unhandled Qt exception; see report.json'
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
