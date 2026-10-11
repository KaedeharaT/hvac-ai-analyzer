"""Native Windows, real WebGL and existing OSM model QA. Outputs are local only."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--model',required=True)
    parser.add_argument('--workspace',default='artifacts/model-viewer-qa')
    parser.add_argument('--scale',default='1'); parser.add_argument('--reopen')
    args=parser.parse_args(); workspace=Path(args.workspace).resolve(); workspace.mkdir(parents=True,exist_ok=True)
    os.environ['BUILDING_AI_CONFIG_DIR']=str(workspace/'config')
    os.environ['QT_QPA_PLATFORM']='windows'; os.environ['QT_SCALE_FACTOR']=args.scale
    from PyQt5.QtCore import Qt,QPoint,QPointF,QEvent,QTimer
    from PyQt5.QtGui import QWheelEvent,QMouseEvent
    from PyQt5.QtWidgets import QApplication,QMessageBox
    from PyQt5.QtTest import QTest
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    from building_ai.ui.model_viewer_backend import prepare_model_viewer
    assert prepare_model_viewer(), "Optional WebEngine unavailable"
    app=QApplication([]); app.setQuitOnLastWindowClosed(False)
    errors=[]
    def dismiss_error():
        popup=app.activeModalWidget()
        if isinstance(popup,QMessageBox): errors.append(popup.text()); print('GUI error:',popup.text(),flush=True); popup.accept()
    watchdog=QTimer(); watchdog.timeout.connect(dismiss_error); watchdog.start(250)
    from building_ai.ui.main_window import MainWindow
    from building_ai.ui.context import ApplicationContext
    from building_ai.config import Settings
    from building_ai.i18n import LanguageManager,tr
    from scripts.capture_product_qa import _context
    def events(ms=100): QTest.qWait(ms); app.processEvents()
    def until(predicate,seconds=30):
        deadline=time.monotonic()+seconds
        while not predicate() and time.monotonic()<deadline: events()
        assert predicate(), 'Timed out waiting for real renderer'
    context=ApplicationContext(Settings.load()) if args.reopen else _context(workspace/'runtime')
    if args.reopen: context.open_project(args.reopen)
    else:
        semantics=context.semantic_result
        source=workspace/'synthetic-bems.csv'; context.dataframe.to_csv(source,index=False)
        context.import_data(str(source)); context.semantic_result=semantics
        context.projects.save_semantics(context.current_project.project_id,semantics.semantic_results)
        context.run_diagnosis(); context.settings.save()
    window=MainWindow(context); window.show(); window.navigate_to('drawing_intelligence'); events()
    drawing=window.page_by_key['drawing_intelligence']; QTest.mouseClick(drawing.preview_button,Qt.LeftButton); events()
    viewer=drawing.model_dialog
    until(lambda:viewer.web_ready or viewer.web_error or viewer.view is None)
    assert viewer.web_ready and not viewer.web_error,viewer.web_error
    def js(code):
        received=[]; viewer.page.runJavaScript(code,received.append); until(lambda:bool(received),10); return received[0]
    if not args.reopen:
        assert viewer.model is None
        assert viewer.status.text()==tr('model_empty')
        viewer.import_path(Path(args.model).resolve()); events(500)
    until(lambda:js('window.viewer.state().elements')>0)
    assert viewer.model
    state=js('window.viewer.state()'); assert state['elements']==len(viewer.model['elements']) and state['unit']=='m'
    if args.reopen:
        assert viewer.events and LanguageManager.instance().language=='ja_JP'
        viewer.grab().save(str(workspace/'reopened.png'))
        (workspace/'reopen.json').write_text(json.dumps({'elements':state['elements'],'language':'ja_JP','association_events':len(viewer.events)}),encoding='utf-8')
        viewer.close(); window.close(); return
    viewer.grab().save(str(workspace/'loaded.png'))
    print('Existing model rendered:',state['elements'],flush=True)
    # Camera gestures must change the real WebGL camera, not source vertices.
    target=viewer.view.focusProxy() or viewer.view
    center=target.rect().center(); before=js('window.viewer.state()')
    QTest.mousePress(target,Qt.LeftButton,pos=center)
    def move_pressed(position,button):
        QApplication.sendEvent(target,QMouseEvent(QEvent.MouseMove,QPointF(position),QPointF(target.mapToGlobal(position)),Qt.NoButton,button,Qt.NoModifier)); events(30)
    for delta in range(5,66,10): move_pressed(center+QPoint(delta,25),Qt.LeftButton)
    QTest.mouseRelease(target,Qt.LeftButton,pos=center+QPoint(65,25)); events()
    rotated=js('window.viewer.state()'); assert rotated['position']!=before['position']
    QTest.mousePress(target,Qt.RightButton,pos=center)
    move_pressed(center+QPoint(25,20),Qt.RightButton); QTest.mouseRelease(target,Qt.RightButton,pos=center+QPoint(25,20)); events()
    assert js('window.viewer.state().target')!=rotated['target']
    before_zoom=js('window.viewer.state().position')
    event=QWheelEvent(QPointF(center),QPointF(target.mapToGlobal(center)),QPoint(),QPoint(0,120),Qt.NoButton,Qt.NoModifier,Qt.NoScrollPhase,False)
    QApplication.sendEvent(target,event); events()
    assert js('window.viewer.state().position')!=before_zoom
    QTest.mouseClick(viewer.reset_button,Qt.LeftButton); events()
    # Click an actual projected face and let the renderer choose the visible object.
    for element in viewer.model['elements']:
        point=js('window.viewer.screenPoint('+json.dumps(element['object_id'])+')')
        if point and 5<point[0]<target.width()-5 and 5<point[1]<target.height()-5:
            QTest.mouseClick(target,Qt.LeftButton,pos=QPoint(round(point[0]),round(point[1]))); events()
            if viewer.selected_object(): break
    assert viewer.selected_object() and js('window.viewer.state().selected')==viewer.selected_object()['object_id']
    assert tr('model_no_link') in viewer.details.toPlainText()
    # Co-planar openings must be selectable instead of always selecting the wall.
    openings=[e for e in viewer.model['elements'] if e['source_type']=='OS:SubSurface']
    if openings:
        for element in openings:
            point=js('window.viewer.screenPoint('+json.dumps(element['object_id'])+')')
            if point and 5<point[0]<target.width()-5 and 5<point[1]<target.height()-5:
                QTest.mouseClick(target,Qt.LeftButton,pos=QPoint(round(point[0]),round(point[1]))); events()
                if viewer.selected_object()['source_type']=='OS:SubSurface': break
        assert viewer.selected_object()['source_type']=='OS:SubSurface'
    category=viewer.categories.item(0); total=js('window.viewer.state().visible')
    category.setCheckState(Qt.Unchecked); events(); assert js('window.viewer.state().visible')<total
    category.setCheckState(Qt.Checked); events(); assert js('window.viewer.state().visible')==total
    viewer.equipment.setCurrentIndex(1); viewer.reviewer.setText('Synthetic GUI QA')
    viewer.note.setText('Software test reference only; not a historical equipment installation claim.')
    viewer.side_scroll.ensureWidgetVisible(viewer.confirm_button); events()
    QTest.mouseClick(viewer.confirm_button,Qt.LeftButton); events()
    assert viewer.current_association(),errors
    print('Camera, selection, filtering and manual association PASS',flush=True)
    export=workspace/'model-provenance.json'
    viewer.service.export_provenance(viewer.project_id,viewer.model['model_id'],export)
    assert json.loads(export.read_text(encoding='utf-8'))['association_events']
    baseline=viewer.service.load(viewer.project_id,viewer.model['model_id'])['elements']
    scientific_result=context.energy_analysis_view_result
    captures=[]
    for language in ('zh_CN','en_US','ja_JP'):
        LanguageManager.instance().set_language(language); context.settings.language=language; events()
        assert context.energy_analysis_view_result is scientific_result
        assert js('window.viewer.state().elements')==len(baseline)
        linked = viewer.current_association()['equipment_id']
        kpi = next(k for k in context.diagnosis_result.analytics.equipment_kpis if k.equipment_id == linked)
        for metric, label in (('cop','analysis_metric_mean_cop'), ('delta_t_c','analysis_metric_mean_delta_t_c')):
            expected = f"{kpi.metric_summary[metric]['mean']:.3g}"
            assert tr(label)+': '+expected in viewer.details.toPlainText()
        for width,height in ((1280,720),(1440,900),(1920,1080)):
            viewer.resize(width,height); events(); viewer.side_scroll.verticalScrollBar().setValue(0)
            viewer.grab().save(str(workspace/f'{language}-{width}x{height}.png'))
            captures.append({'locale':language,'size':[viewer.width(),viewer.height()],'dpr':viewer.devicePixelRatioF()})
        viewer.side_scroll.ensureWidgetVisible(viewer.details); viewer.details.verticalScrollBar().setValue(viewer.details.verticalScrollBar().maximum())
        events(); viewer.grab().save(str(workspace/f'{language}-equipment-details.png'))
    assert baseline==viewer.service.load(viewer.project_id,viewer.model['model_id'])['elements']
    context.settings.save(); project_id=context.current_project.project_id
    viewer.side_scroll.ensureWidgetVisible(viewer.open_equipment); events()
    QTest.mouseClick(viewer.open_equipment,Qt.LeftButton); events()
    assert window.stack.currentWidget() is window.page_by_key['equipment']
    window.close(); events()
    completed=subprocess.run([sys.executable,__file__,'--model',args.model,'--workspace',str(workspace),'--scale',args.scale,'--reopen',project_id],capture_output=True,text=True,encoding='utf-8',timeout=120)
    assert completed.returncode==0,completed.stdout+completed.stderr
    assert not errors,errors
    report={'renderer':'Three.js / native QtWebEngine WebGL','model_hash':viewer.model['model_hash'],
        'elements':len(baseline),'unit':'m','camera_gestures':'PASS','raycast_selection':'PASS',
        'category_filter':'PASS','explicit_reference':'PASS','restart':'PASS','captures':captures}
    (workspace/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
