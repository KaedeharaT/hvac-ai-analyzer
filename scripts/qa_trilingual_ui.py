"""Exercise the real desktop UI; all outputs stay in gitignored artifacts.

Run once per --scale (1, 1.25, 1.5). --native uses the Windows platform.
--live-llm uses only an already configured LOCAL provider on synthetic data.
Optional --drawing / --model run actual inference; assets are never committed.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scale',default='1')
    parser.add_argument('--native',action='store_true')
    parser.add_argument('--live-llm',action='store_true')
    parser.add_argument('--drawing'); parser.add_argument('--model')
    parser.add_argument('--reopen'); parser.add_argument('--workspace')
    args=parser.parse_args()
    os.environ['QT_SCALE_FACTOR']=args.scale
    os.environ['QT_AUTO_SCREEN_SCALE_FACTOR']='0'
    os.environ['QT_QPA_PLATFORM']='windows' if args.native else 'offscreen'
    from PyQt5.QtCore import Qt, QPoint
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication, QScrollArea
    from building_ai.config import Settings
    from building_ai.ui.context import ApplicationContext
    from building_ai.ui.main_window import MainWindow
    from building_ai.i18n import LanguageManager
    from building_ai.knowledge import KnowledgeService
    from building_ai.knowledge_catalog import source_registry, load_materialized_facts, CATALOG_DIR
    import pandas as pd

    original=Settings.load() if args.live_llm else None
    workspace=Path(args.workspace) if args.workspace else ROOT/'artifacts'/'trilingual_qa'/('scale-'+args.scale)
    workspace.mkdir(parents=True,exist_ok=True)
    os.environ['BUILDING_AI_CONFIG_DIR']=str(workspace/'config')
    QApplication.setAttribute(Qt.AA_DontUseNativeDialogs,True)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling,True)
    app=QApplication([]); app.setQuitOnLastWindowClosed(False)
    from building_ai.ui.fonts import configure_fonts
    configure_fonts()
    font=app.font().family()
    failures=[]
    def exception(kind,value,tb):
        import traceback
        failures.append(''.join(traceback.format_exception(kind,value,tb)))
        print(failures[-1],flush=True)
    sys.excepthook=exception
    def events(ms=150): QTest.qWait(ms); app.processEvents()
    def click(widget):
        parent=widget.parentWidget()
        while parent:
            if isinstance(parent,QScrollArea): parent.ensureWidgetVisible(widget); events(20)
            parent=parent.parentWidget()
        if args.native: QTest.mouseClick(widget,Qt.LeftButton)
        else: widget.click()  # Offscreen rendering has no native input hit-testing.
        events()
    if args.reopen:
        context=ApplicationContext(Settings.load()); context.open_project(args.reopen)
        window=MainWindow(context); window.show(); events()
        assert LanguageManager.instance().language=='ja_JP'
        assert context.dataframe is not None and context.equipment
        window.grab().save(str(workspace/'reopened-japanese.png'))
        window.close()
        (workspace/'reopen.json').write_text(json.dumps({'locale':'ja_JP','rows':len(context.dataframe),'equipment':len(context.equipment)}),encoding='utf-8')
        return 0

    settings=Settings(provider='not_configured',language='en_US',data_dir=workspace/'runtime')
    context=ApplicationContext(settings); project=context.projects.create('Synthetic BEMS QA')
    context.current_project=project
    window=MainWindow(context); window.show(); events()
    def navigate(key): click(window.navigation_buttons[next(i for i,(_,k,_) in enumerate(window.NAVIGATION) if k==key)])
    navigate('settings')
    language=window.page_by_key['settings'].language
    language.setCurrentIndex(language.findData('ja_JP')); events()
    # Persist real BEMS through the production import service; use UI for review and diagnosis.
    frame=pd.DataFrame({'timestamp':pd.date_range('2026-07-01',periods=96,freq='15min'),
        'AHP-01 往水温度 (°C)':[6.0+i%8*.03 for i in range(96)],
        'AHP-01 返水温度 (°C)':[8.1+i%8*.04 for i in range(96)],
        'AHP-01 流量 (m3/h)':[18+i%6*.2 for i in range(96)],
        'AHP-01 消費電力 (kW)':[42+i%12*1.1 for i in range(96)]})
    source=workspace/'synthetic-bems.csv'; frame.to_csv(source,index=False)
    navigate('import_data'); context.import_data(str(source.resolve())); window.refresh_pages(); events()
    assert len(context.dataframe)==96
    print("Real BEMS import PASS",flush=True)
    navigate('semantic_mapping'); click(window.page_by_key['semantic_mapping'].run_button)
    semantics=window.page_by_key['semantic_mapping']
    # Confirm the known meanings of this synthetic fixture through the real review controls.
    labels=('heat_source_supply_temp','heat_source_return_temp','heat_source_flow','heat_source_power')
    for name,label in zip(frame.columns[1:],labels):
        row=next(i for i,p in enumerate(context.semantic_result.semantic_results) if p.raw_name==name)
        semantics.table.selectRow(row); events()
        semantics.label.setCurrentText(label); semantics.equipment_id.setText('AHP-01')
        semantics.note.setText('Synthetic QA input: manually confirmed meaning and equipment.')
        click(semantics.accept)
    assert context.equipment
    navigate('analysis'); click(window.page_by_key['analysis'].run_button)
    deadline=time.monotonic()+120
    while window.page_by_key['analysis']._analysis_worker is not None and time.monotonic()<deadline: events(100)
    assert context.diagnosis_result is not None
    assert context.diagnosis_result.analytics.available_kpis
    print("Real semantic review and diagnosis PASS",flush=True)
    KnowledgeService(context.database).ingest_catalog(source_registry(),load_materialized_facts(CATALOG_DIR))
    navigate('knowledge_base'); knowledge=window.page_by_key['knowledge_base']; knowledge.query.setText('COP'); click(knowledge.search_button)
    assert knowledge._results
    print("Knowledge search PASS",flush=True)

    ai_status='NOT_REQUESTED'
    if original:
        assert original.provider=='local_llm', 'Live QA supports local providers only'
        settings.provider=original.provider; settings.model=original.model; settings.ollama_url=original.ollama_url
        context.reload_llm(); navigate('agent'); agent=window.page_by_key['agent']
        assert agent.input.isEnabled()
        agent.input.setPlainText('一般的なHVACの保守点検について、日本語で簡潔に説明してください。'); click(agent.send_button)
        deadline=time.monotonic()+240
        while agent._agent_worker is not None and time.monotonic()<deadline: events(200)
        assert agent._agent_worker is None, 'Local LLM timeout'
        from building_ai.ui.agent_chat import ChatMessage
        answers=[w.raw_text for w in agent.transcript.items if isinstance(w,ChatMessage) and w.role=='assistant' and w.raw_text]
        if answers: (workspace/'local-ai-answer.txt').write_text(answers[-1],encoding='utf-8')
        assert answers and any('\u3040'<=c<='\u30ff' for c in answers[-1])
        with context.database.connect() as connection:
            traces=[json.loads(row['payload']) for row in connection.execute('SELECT payload FROM agent_traces')]
        assert any(call.get('status')=='SUCCEEDED' for trace in traces for call in trace.get('llm_calls',[])), 'Only fallback text was returned'
        ai_status='PASS'; (workspace/'local-ai-answer.txt').write_text(answers[-1],encoding='utf-8')
        print('Live Japanese assistant PASS',flush=True)

    spatial='NOT_REQUESTED'
    if args.drawing and args.model:
        settings.drawing_model_path=args.model
        navigate('drawing_intelligence'); drawing=window.page_by_key['drawing_intelligence']
        drawing.select_drawing(context.drawings.import_drawing(project.project_id,Path(args.drawing).resolve())); drawing.refresh(); events(); click(drawing.detect_button)
        assert drawing.rows
        row_index=next(i for i,r in enumerate(drawing.rows) if r['class_name']=='aircon')
        drawing.list.setCurrentRow(row_index); drawing.class_choice.setCurrentText('aircon'); click(drawing.confirm)
        drawing.map_equipment.setCurrentIndex(1); click(drawing.map_button)
        assert drawing.selected()['review_status']=='confirmed' and drawing.selected()['equipment_id']
        before=context.drawings.image_positions(project.project_id,drawing.drawing['drawing_id'])
        click(drawing.zoom_in); click(drawing.zoom_out); click(drawing.fit)
        QTest.mousePress(drawing.viewer.viewport(),Qt.LeftButton,pos=drawing.viewer.viewport().rect().center())
        QTest.mouseMove(drawing.viewer.viewport(),drawing.viewer.viewport().rect().center()+QPoint(20,20))
        QTest.mouseRelease(drawing.viewer.viewport(),Qt.LeftButton); events()
        assert before==context.drawings.image_positions(project.project_id,drawing.drawing['drawing_id'])
        destination=(workspace/'image-positions.json').resolve()
        context.drawings.export_image_positions(project.project_id,drawing.drawing["drawing_id"],destination)
        assert json.loads(destination.read_text(encoding='utf-8'))==before
        click(drawing.open_equipment)
        assert window.stack.currentWidget() is window.page_by_key['equipment']
        spatial='PASS'; print('Real drawing workflow PASS',flush=True)

    captures=[]; sizes=((1280,720),(1440,900),(1920,1080))
    for locale in ('zh_CN','en_US','ja_JP'):
        language.setCurrentIndex(language.findData(locale)); events()
        for width,height in sizes:
            window.resize(width,height); events()
            for key in ('dashboard','import_data','semantic_mapping','equipment','energy_analysis','analysis','drawing_intelligence','agent','knowledge_base','settings'):
                navigate(key)
                if key=='analysis': window.page_by_key[key].tabs.setCurrentIndex(2); events()
                target=workspace/f'{locale}-{width}x{height}-{key}.png'
                assert window.grab().save(str(target))
                captures.append({'locale':locale,'page':key,'requested':[width,height],'actual':[window.width(),window.height()],'dpr':window.devicePixelRatioF()})
        print('Captured '+locale,flush=True)
    settings.save(); window.close(); events()
    command=[sys.executable,__file__,'--scale',args.scale,'--workspace',str(workspace),'--reopen',project.project_id]
    if args.native: command.append('--native')
    completed=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',timeout=120)
    assert completed.returncode==0,completed.stderr
    report={'platform':app.platformName(),'scale':args.scale,'font':font,'ai':ai_status,'spatial':spatial,
            'rows':len(context.dataframe),'equipment':len(context.equipment),'persistence':'PASS','captures':captures,'exceptions':failures}
    (workspace/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    assert not failures,failures
    print(json.dumps({k:v for k,v in report.items() if k!='captures'}),flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
