from __future__ import annotations
import base64
import pytest
from building_ai.services.agent_service import AgentService
from building_ai.services.drawing_service import DrawingService
from building_ai.storage import Database, ProjectStore, TimeseriesStore
from building_ai.vision import DrawingDetection, DrawingModelInfo, DrawingModelRegistry, FakeDrawingDetector, YOLODrawingDetector

PNG=base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScL9aQAAAABJRU5ErkJggg==")
def make_fixture(tmp_path):
    db=Database(tmp_path/'db.sqlite'); projects=ProjectStore(db); first=projects.create('One'); second=projects.create('Two'); image=tmp_path/'plan.png'; image.write_bytes(PNG); return projects,DrawingService(db,tmp_path/'data'),first,second,image
def sample(): return DrawingDetection('aircon',.92,1,2,20,30,100,80)
def test_detector_schema_and_fake():
    assert FakeDrawingDetector([sample()]).detect('ignored')[0].normalized_class=='air_conditioning_unit'
def test_missing_model_is_optional():
    detector=YOLODrawingDetector(DrawingModelInfo('legacy','Legacy','ultralytics',''))
    assert detector.is_available() is False
    with pytest.raises(RuntimeError): detector.load_model()
def test_import_persistence_review_mapping_and_isolation(tmp_path):
    projects,service,first,second,image=make_fixture(tmp_path); drawing=service.import_drawing(first.project_id,image); assert drawing['managed_path'] != str(image)
    rows=service.save_detections(first.project_id,drawing['drawing_id'],'fake',[sample()]); detection=rows[0]; assert len(service.list_detections(first.project_id))==1; assert service.list_detections(second.project_id)==[]
    service.review_detection(first.project_id,detection['detection_id'],'confirmed','aircon'); service.map_equipment(first.project_id,detection['detection_id'],'AHP-3-3')
    assert service.equipment_location(first.project_id,'AHP-3-3')[0]['review_status']=='confirmed'
    service.review_detection(first.project_id,detection['detection_id'],'rejected')
    assert service.list_detections(first.project_id)[0]['review_status']=='rejected'
def test_rejected_detection_cannot_map(tmp_path):
    projects,service,first,_,image=make_fixture(tmp_path); drawing=service.import_drawing(first.project_id,image); row=service.save_detections(first.project_id,drawing['drawing_id'],'fake',[sample()])[0]
    with pytest.raises(ValueError): service.map_equipment(first.project_id,row['detection_id'],'AHP')
def test_read_only_agent_tools_and_unknown_mapping_abstain(tmp_path):
    projects,service,first,_,image=make_fixture(tmp_path); drawing=service.import_drawing(first.project_id,image); row=service.save_detections(first.project_id,drawing['drawing_id'],'fake',[sample()])[0]; service.review_detection(first.project_id,row['detection_id'],'confirmed')
    agent=AgentService(projects,TimeseriesStore(tmp_path/'series'),drawings=service)
    assert agent.tools.call('get_drawing_summary',project_id=first.project_id).data['drawings']==1
    unknown=agent.tools.call('get_equipment_drawing_location',project_id=first.project_id,equipment_id='AHP-3-3').data
    assert unknown['reliable'] is False and 'No confirmed' in unknown['reason']
    assert 'review_detection' not in agent.tools.names() and 'map_equipment' not in agent.tools.names()
