"""Real Qt input, WebGL gestures, inference and configured local LLM QA."""
from __future__ import annotations

import json
import time
from pathlib import Path

from PyQt5.QtCore import Qt, QPoint, QPointF, QEvent
from PyQt5.QtGui import QMouseEvent, QWheelEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from building_ai.i18n import tr


def wait_for(predicate, events, timeout=90):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        events(100)
    assert predicate(), 'Native workflow timed out'


def javascript(viewer, events, source):
    result = []
    viewer.page.runJavaScript(source, result.append)
    wait_for(lambda: bool(result), events, 15)
    return result[0]


def exercise_independent_assets(window, context, manifest, workspace, report,
                                navigate, click, events, answers, reopen=False):
    """Load drawing-only and model-only projects through the native GUI."""
    for kind in ('drawing', 'model'):
        navigate('projects'); projects = window.page_by_key['projects']
        if reopen:
            project_id = report['independent_assets'][kind]['project_id']
            row = next(i for i in range(projects.list.rowCount())
                       if projects.list.item(i, 0).data(Qt.UserRole) == project_id)
            projects.list.selectRow(row); click(projects.open_button)
        else:
            answers.append('SYNTHETIC TEST identity: independent ' + kind)
            click(projects.create_button)
        assert context.dataframe is None and not context.equipment
        navigate('drawing_intelligence'); drawing = window.page_by_key['drawing_intelligence']
        if kind == 'drawing':
            if not reopen:
                answers.append(manifest['drawings'][0]); click(drawing.import_button)
            assert drawing.pix and not drawing.pix.pixmap().isNull()
            click(drawing.zoom_in); click(drawing.fit)
        else:
            assert drawing.drawing is None
            click(drawing.preview_button); viewer = drawing.model_dialog
            wait_for(lambda: viewer.web_ready, events)
            if not reopen:
                assert viewer.model is None
                answers.append(manifest['model']); click(viewer.import_button)
            wait_for(lambda: javascript(viewer, events, 'window.viewer.state().elements') > 0, events)
            assert not viewer.events
            viewer.grab().save(str(workspace / ('independent-model' + ('-reopened' if reopen else '') + '.png')))
            viewer.close(); events()
        window.grab().save(str(workspace / ('independent-' + kind + '.png')))
        report.setdefault('independent_assets', {})[kind] = {
            'project_id': context.current_project.project_id,
            'without_bems': 'PASS', 'reopened': reopen,
        }


def restore_spatial(window, context, workspace, navigate, click, events):
    navigate('drawing_intelligence')
    drawing = window.page_by_key['drawing_intelligence']
    assert drawing.drawing and drawing.rows
    saved = json.loads((workspace / 'image-positions-0.json').read_text(encoding='utf-8'))
    index = next(i for i in range(drawing.drawing_choice.count()) if drawing.drawing_choice.itemData(i)['drawing_id'] == saved[0]['drawing_id'])
    drawing.drawing_choice.setCurrentIndex(index); events()
    assert saved == context.drawings.image_positions(context.current_project.project_id, drawing.drawing['drawing_id'])
    click(drawing.preview_button)
    viewer = drawing.model_dialog
    wait_for(lambda: viewer.web_ready, events)
    wait_for(lambda: javascript(viewer, events, 'window.viewer.state().elements') > 0, events)
    assert viewer.events
    viewer.grab().save(str(workspace / 'reopened-model.png'))
    result = {'projects': 'PASS', 'manual_reviews': 'PASS', 'drawings': 'PASS', 'model': 'PASS', 'elements': len(viewer.model['elements']), 'association_events': len(viewer.events)}
    viewer.close(); events()
    return result


def exercise(window, context, manifest, workspace, report, navigate, click, events, answers, save):
    # This final project is explicitly synthetic. References to historical
    # drawing/model assets are QA records, never real installation assertions.
    assert context.current_project.name.startswith('SYNTHETIC')
    navigate('drawing_intelligence')
    drawing = window.page_by_key['drawing_intelligence']
    answers.append(manifest['weights']); click(drawing.model_button)
    report['drawings'] = []
    for index, path in enumerate(manifest['drawings']):
        answers.append(path); click(drawing.import_button)
        assert drawing.pix and not drawing.pix.pixmap().isNull()
        click(drawing.detect_button)
        record = {'drawing_id': drawing.drawing['drawing_id'], 'predictions': len(drawing.rows), 'classes': sorted({r['class_name'] for r in drawing.rows})}
        assert drawing.rows, 'Actual detector returned no predictions; review was not exercised'
        row = next((i for i, item in enumerate(drawing.rows) if item['class_name'] == 'aircon'), 0)
        drawing.list.setCurrentRow(row); drawing.class_choice.setCurrentText(drawing.selected()['class_name'])
        click(drawing.confirm)
        drawing.map_equipment.setCurrentIndex(1); click(drawing.map_button)
        assert drawing.selected()['review_status'] == 'confirmed' and drawing.selected()['equipment_id']
        confirmed_id = drawing.selected()['detection_id']
        if len(drawing.rows) > 1:
            drawing.list.setCurrentRow(next(i for i, item in enumerate(drawing.rows) if item['detection_id'] != confirmed_id)); click(drawing.reject)
            assert drawing.selected()['review_status'] == 'rejected'
        before = context.drawings.image_positions(context.current_project.project_id, drawing.drawing['drawing_id'])
        click(drawing.zoom_in); click(drawing.zoom_out); click(drawing.fit)
        target = drawing.viewer.viewport(); center = target.rect().center()
        QTest.mousePress(target, Qt.LeftButton, pos=center)
        QApplication.sendEvent(target, QMouseEvent(QEvent.MouseMove, QPointF(center + QPoint(30, 20)), Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
        QTest.mouseRelease(target, Qt.LeftButton, pos=center + QPoint(30, 20)); events()
        assert before == context.drawings.image_positions(context.current_project.project_id, drawing.drawing['drawing_id'])
        export = workspace / f'image-positions-{index}.json'
        answers.append(str(export)); click(drawing.export_button)
        assert json.loads(export.read_text(encoding='utf-8')) == before
        drawing.list.setCurrentRow(next(i for i, item in enumerate(drawing.rows) if item['detection_id'] == confirmed_id)); click(drawing.open_equipment)
        assert window.stack.currentWidget() is window.page_by_key['equipment']
        navigate('drawing_intelligence')
        window.grab().save(str(workspace / f'drawing-{index}.png'))
        record.update(review='PASS', rejected='PASS', linkage='QA reference only', export='PASS')
        report['drawings'].append(record); save()
    click(drawing.preview_button)
    viewer = drawing.model_dialog
    wait_for(lambda: viewer.web_ready, events)
    assert viewer.model is None
    answers.append(manifest['model']); click(viewer.import_button)
    js = lambda source: javascript(viewer, events, source)
    wait_for(lambda: js('window.viewer.state().elements') > 0, events)
    target = viewer.view.focusProxy() or viewer.view
    center = target.rect().center()
    before = js('window.viewer.state()')
    def drag(button, offset):
        QTest.mousePress(target, button, pos=center)
        for fraction in range(1, 6):
            position = center + offset * fraction
            QApplication.sendEvent(target, QMouseEvent(QEvent.MouseMove, QPointF(position), QPointF(target.mapToGlobal(position)), Qt.NoButton, button, Qt.NoModifier)); events(30)
        QTest.mouseRelease(target, button, pos=position); events()
    drag(Qt.LeftButton, QPoint(13, 5))
    assert js('window.viewer.state().position') != before['position']
    before_pan = js('window.viewer.state().target'); drag(Qt.RightButton, QPoint(5, 4))
    assert js('window.viewer.state().target') != before_pan
    before_zoom = js('window.viewer.state().position')
    QApplication.sendEvent(target, QWheelEvent(QPointF(center), QPointF(target.mapToGlobal(center)), QPoint(), QPoint(0, 120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)); events()
    assert js('window.viewer.state().position') != before_zoom
    click(viewer.reset_button)
    for element in viewer.model['elements']:
        point = js('window.viewer.screenPoint(' + json.dumps(element['object_id']) + ')')
        if point and 5 < point[0] < target.width() - 5 and 5 < point[1] < target.height() - 5:
            QTest.mouseClick(target, Qt.LeftButton, pos=QPoint(round(point[0]), round(point[1]))); events()
            if viewer.selected_object(): break
    assert viewer.selected_object()
    assert tr('model_no_link') in viewer.details.toPlainText()
    total = js('window.viewer.state().visible')
    category = viewer.categories.item(0)
    category.setCheckState(Qt.Unchecked); events()
    assert js('window.viewer.state().visible') < total
    category.setCheckState(Qt.Checked); events()
    assert js('window.viewer.state().visible') == total
    viewer.equipment.setCurrentIndex(1)
    viewer.reviewer.setText('LOCAL SOFTWARE QA')
    viewer.note.setText('QA ONLY: unrelated synthetic BEMS reference; no real equipment location or installation claim.')
    click(viewer.confirm_button)
    assert viewer.current_association()
    assert tr('analysis_metric_mean_cop') in viewer.details.toPlainText()
    answers.append(str(workspace / 'model-provenance.json')); click(viewer.export_button)
    assert (workspace / 'model-provenance.json').is_file()
    report['model'] = {'format': viewer.model['source_format'], 'version': viewer.model['source_version'],
                       'model_hash': viewer.model['model_hash'], 'unit': viewer.model['unit'], 'elements': len(viewer.model['elements']),
                       'rotation': 'PASS', 'zoom': 'PASS', 'pan': 'PASS', 'selection': 'PASS', 'filter': 'PASS', 'linkage': 'QA reference only'}
    viewer.grab().save(str(workspace / 'model.png'))
    click(viewer.open_equipment); events()
    assert window.stack.currentWidget() is window.page_by_key['equipment']
    save()
    navigate('knowledge_base')
    # Install the bundled, attributed public catalog in this isolated database.
    from building_ai.knowledge import KnowledgeService
    from building_ai.knowledge_catalog import source_registry, load_materialized_facts, CATALOG_DIR
    KnowledgeService(context.database).ingest_catalog(source_registry(), load_materialized_facts(CATALOG_DIR))
    knowledge = window.page_by_key['knowledge_base']; knowledge.query.setText('COP'); click(knowledge.search_button)
    assert knowledge._results
    report['knowledge'] = {'results': len(knowledge._results), 'query': 'COP'}
    window.grab().save(str(workspace / 'knowledge.png'))
    exercise_ai(window, context, manifest, workspace, report, navigate, click, events, save)
    capture_languages(window, context, workspace, report, navigate, events, save)


def capture_languages(window, context, workspace, report, navigate, events, save):
    baseline = dict(context.energy_analysis_result.summary)
    report['languages'] = []
    for locale in ('zh_CN', 'en_US', 'ja_JP'):
        navigate('settings'); language = window.page_by_key['settings'].language
        language.setCurrentIndex(language.findData(locale)); events()
        for size in ((1280, 720), (1440, 900), (1920, 1080)):
            window.resize(*size); events()
            for _, key, _ in window.NAVIGATION:
                navigate(key)
                window.grab().save(str(workspace / f'{locale}-{size[0]}-{key}.png'))
        assert dict(context.energy_analysis_result.summary) == baseline
        report['languages'].append(locale); save()
    context.settings.save()


def exercise_ai(window, context, manifest, workspace, report, navigate, click, events, save):
    report['ai'] = []
    original_project = context.current_project.project_id
    def select_project(name=None, project_id=None):
        previous_project = context.current_project.project_id
        navigate('projects'); page = window.page_by_key['projects']
        row = next(i for i in range(page.list.rowCount()) if (page.list.item(i, 0).text() == name if name else page.list.item(i, 0).data(Qt.UserRole) == project_id))
        page.list.selectRow(row); click(page.open_button)
        semantic = window.page_by_key['semantic_mapping']
        if context.current_project.project_id != previous_project:
            assert not semantic.equipment_id.text() and not semantic.details.toPlainText(), 'Stale semantic review from another project'
        if semantic.table.rowCount():
            semantic.table.selectRow(0); events()
            assert semantic.equipment_id.text() == (semantic.current_result().effective_equipment_id or '')
    for item in manifest.get('questions', []):
        if isinstance(item, dict):
            select_project(name=item['project'])
            question = item['question']
        else: question = item
        navigate('agent'); agent = window.page_by_key['agent']
        agent.input.setPlainText(question); click(agent.send_button)
        wait_for(lambda: agent._agent_worker is None, events, 300)
        from building_ai.ui.agent_chat import ChatMessage
        messages = [w.raw_text for w in agent.transcript.items if isinstance(w, ChatMessage) and w.role == 'assistant' and w.raw_text]
        assert messages
        report['ai'].append({'question': question, 'project_id': context.current_project.project_id, 'answer': messages[-1]})
        with context.database.connect() as conn:
            traces = [json.loads(row['payload']) for row in conn.execute('SELECT payload FROM agent_traces')]
        report['traces'] = traces
        save(); print('Live assistant response', len(report['ai']), flush=True)
    if context.current_project.project_id != original_project:
        select_project(project_id=original_project)
