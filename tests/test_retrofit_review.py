"""Synthetic identity evidence never certifies any historical building."""
import pickle
from dataclasses import asdict
import pandas as pd
import pytest
from PIL import Image
from building_ai.config import Settings
from building_ai.ui.context import ApplicationContext
from building_ai.vision import DrawingDetection
from building_ai.agent_runtime import AgentRuntime
from building_ai.services.retrofit_review_renderer import render_review


@pytest.fixture
def example(tmp_path,monkeypatch):
    monkeypatch.setenv('BUILDING_AI_CONFIG_DIR',str(tmp_path/'config'))
    context=ApplicationContext(Settings(data_dir=tmp_path/'data'))
    project=context.projects.create('SYNTHETIC correspondence fixture'); context.open_project(project.project_id)
    source=tmp_path/'synthetic.csv'
    pd.DataFrame({'time':pd.date_range('2025-01-01',periods=96,freq='15min'),
                  'CH-1 LWT °C':[6.]*96,'CH-1 EWT °C':[11.]*96,
                  'CH-1 Flow m3/h':[18.]*96,'CH-1 Power kW':[100.]*96}).to_csv(source,index=False)
    context.import_data(str(source)); context.run_semantics(); context.run_diagnosis()
    image=tmp_path/'synthetic-drawing.png'; Image.new('RGB',(400,200),'white').save(image)
    drawing=context.drawings.import_drawing(project.project_id,image)
    row=context.drawings.save_detections(project.project_id,drawing['drawing_id'],'synthetic',[
        DrawingDetection('aircon',.9,20,40,80,100,400,200)])[0]
    return context,project,row,drawing


def test_predictions_and_test_links_do_not_complete_identity_evidence(example):
    ctx,project,row,drawing=example; equipment=ctx.equipment[0].equipment_id
    report=ctx.agent.get_retrofit_review(project.project_id)
    assert report['drawing_evidence']['predicted_count']==1
    assert report['stages']['verified_associations']==0 and report['workflow_status']=='partial'
    assert report['layers']['operation']['items']
    ctx.drawings.review_detection(project.project_id,row['detection_id'],'confirmed','aircon')
    ctx.drawings.map_equipment(project.project_id,row['detection_id'],equipment,association_status='test',reviewer='QA',evidence_note='synthetic reference only')
    report=ctx.agent.get_retrofit_review(project.project_id)
    assert report['stages']['associated_objects']==1 and report['stages']['verified_associations']==0
    assert not ctx.agent.get_equipment_drawing_location(project.project_id,'CH-1')['reliable']
    assert not report['design_ready'] and report['estimated_savings'] is None
    assert report['layers']['building_retrofit']['origin']=='general_review_checklist'


def test_verified_identity_requires_evidence_and_survives_restart(example):
    ctx,project,row,drawing=example; equipment=ctx.equipment[0].equipment_id
    ctx.drawings.review_detection(project.project_id,row['detection_id'],'confirmed')
    with pytest.raises(ValueError,match='reviewer'):
        ctx.drawings.map_equipment(project.project_id,row['detection_id'],equipment,association_status='verified')
    ctx.drawings.map_equipment(project.project_id,row['detection_id'],equipment,association_status='verified',reviewer='Synthetic test author',evidence_note='This synthetic fixture defines CH-1 correspondence; no real building claim.')
    report=ctx.agent.get_retrofit_review(project.project_id,'CH-1')
    assert report['workflow_status']=='evidence_linked' and report['stages']['verified_associations']==1
    assert report['bems_evidence']['imports'][0]['file_hash']
    assert all(x['evidence'] and x['estimated_savings'] is None for x in report['layers']['operation']['items'])
    reopened=ApplicationContext(ctx.settings); reopened.open_project(project.project_id)
    assert reopened.agent.get_retrofit_review(project.project_id)['stages']['verified_associations']==1
    ctx.drawings.review_detection(project.project_id,row['detection_id'],'rejected')
    assert ctx.agent.get_retrofit_review(project.project_id)['stages']['verified_associations']==0


def test_read_model_languages_and_agent_do_not_change_calculations(example):
    ctx,project,row,drawing=example
    before=pickle.dumps(asdict(ctx.diagnosis_result)); energy=pickle.dumps(asdict(ctx.energy_analysis_result))
    report=ctx.agent.get_retrofit_review(project.project_id)
    for locale in ('zh_CN','en_US','ja_JP'):
        text=render_review(report,locale)
        assert 'CH-1' in text and 'retrofit_' not in text
    answer=AgentRuntime(ctx).run('请用中文回答当前项目的既有建筑节能改造建议',project.project_id)
    assert 'get_retrofit_review' in answer.tools_used
    assert '第一层' in answer.answer and '第二层' in answer.answer and '第三层' in answer.answer
    assert pickle.dumps(asdict(ctx.diagnosis_result))==before and pickle.dumps(asdict(ctx.energy_analysis_result))==energy
    other=ctx.projects.create('SYNTHETIC other'); ctx.open_project(other.project_id)
    isolated=ctx.agent.get_retrofit_review(other.project_id)
    assert not isolated['layers']['operation']['items'] and not isolated['drawing_evidence']['associations']


def test_changed_source_excluded_and_other_project_cannot_confirm(example):
    ctx,project,row,drawing=example; equipment=ctx.equipment[0].equipment_id
    other=ctx.projects.create('SYNTHETIC unrelated')
    with pytest.raises(KeyError): ctx.drawings.map_equipment(other.project_id,row['detection_id'],equipment)
    ctx.drawings.review_detection(project.project_id,row['detection_id'],'confirmed')
    ctx.drawings.map_equipment(project.project_id,row['detection_id'],equipment,association_status='verified',reviewer='QA',evidence_note='Synthetic identity evidence')
    Image.new('RGB',(400,200),'black').save(drawing['managed_path'])
    report=ctx.agent.get_retrofit_review(project.project_id)
    assert report['drawing_evidence']['invalid_drawings']==[drawing['drawing_id']]
    assert report['stages']['verified_associations']==0 and report['workflow_status']=='partial'
    assert not ctx.agent.get_equipment_drawing_location(project.project_id,'CH-1')['reliable']


@pytest.mark.parametrize('question,title',[
    ('请用中文回答当前项目的既有建筑节能改造建议','既有建筑节能建议'),
    ('Existing-building energy advice. Please answer in English.','Existing-building energy advice'),
    ('既存建築物の省エネルギー改善提案。日本語で説明してください。','既存建築物の省エネルギー改善提案'),
])
def test_review_agent_uses_explicit_language(example,question,title):
    ctx,project,_,_=example
    answer=AgentRuntime(ctx).run(question,project.project_id)
    assert answer.tools_used==['get_retrofit_review']
    assert answer.answer.startswith(title)
