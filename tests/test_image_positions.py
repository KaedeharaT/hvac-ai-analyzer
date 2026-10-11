import hashlib
import json
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PIL import Image
from building_ai.services.drawing_service import DrawingService
from building_ai.storage import Database, ProjectStore
from building_ai.vision import DrawingDetection


@pytest.fixture
def spatial(tmp_path):
    db=Database(tmp_path/'db.sqlite'); projects=ProjectStore(db)
    first=projects.create('Synthetic A'); second=projects.create('Synthetic B')
    source=tmp_path/'synthetic.png'; Image.new('RGB',(400,200),'white').save(source)
    service=DrawingService(db,tmp_path/'data'); drawing=service.import_drawing(first.project_id,source)
    rows=service.save_detections(first.project_id,drawing['drawing_id'],'fixture',[
        DrawingDetection('aircon',.9,20,40,80,100,400,200),
        DrawingDetection('aircon',.8,200,40,260,100,400,200)])
    return service,first,second,drawing,rows,source


def test_only_confirmed_image_centers_export_with_provenance(spatial,tmp_path):
    service,a,b,d,rows,source=spatial
    assert service.image_positions(a.project_id,d['drawing_id'])==[]
    service.review_detection(a.project_id,rows[0]['detection_id'],'confirmed','window')
    record=service.image_positions(a.project_id,d['drawing_id'])[0]
    assert (record['x'],record['y'],record['unit'])==(50,70,'pixel')
    assert record['coordinate_system']=='image_relative'
    assert record['equipment_id'] is None and record['calibration_reference'] is None
    assert record['model_version'] is None
    assert record['source_image_hash']==hashlib.sha256(source.read_bytes()).hexdigest()
    assert record['cv_prediction']=='aircon' and record['human_class']=='window'
    service.map_equipment(a.project_id,rows[0]['detection_id'],'explicit-human-association')
    dest=tmp_path/'positions.json'; service.export_image_positions(a.project_id,d['drawing_id'],dest)
    assert json.loads(dest.read_text())[0]['equipment_id']=='explicit-human-association'
    with pytest.raises(KeyError): service.image_positions(b.project_id,d['drawing_id'])
    with pytest.raises(KeyError): service.review_detection(b.project_id,rows[0]['detection_id'],'confirmed')
    service.review_detection(a.project_id,rows[0]['detection_id'],'rejected')
    assert service.image_positions(a.project_id,d['drawing_id'])==[]


def test_resized_or_changed_source_is_rejected(spatial):
    service,a,b,d,rows,source=spatial
    service.review_detection(a.project_id,rows[0]['detection_id'],'confirmed')
    Image.new('RGB',(200,100),'white').save(d['managed_path'])
    with pytest.raises(ValueError,match='changed'): service.image_positions(a.project_id,d['drawing_id'])


def test_invalid_bbox_or_inference_dimensions_are_rejected(spatial):
    service,a,b,d,rows,source=spatial
    for detection in [DrawingDetection('aircon',.9,20,40,80,100,200,100),
                      DrawingDetection('aircon',.9,-1,40,80,100,400,200)]:
        row=service.save_detections(a.project_id,d['drawing_id'],'fixture',[detection])[0]
        service.review_detection(a.project_id,row['detection_id'],'confirmed')
        with pytest.raises(ValueError,match='dimensions'): service.image_positions(a.project_id,d['drawing_id'])


def test_equipment_association_cannot_cross_projects(spatial):
    service,a,b,d,rows,source=spatial
    with service.database.connect() as connection:
        connection.execute('INSERT INTO equipment VALUES (?, ?, ?)', ('other-project-equipment', b.project_id, '{}'))
    service.review_detection(a.project_id, rows[0]['detection_id'], 'confirmed')
    with pytest.raises(ValueError, match='another project'):
        service.map_equipment(a.project_id, rows[0]['detection_id'], 'other-project-equipment')
    assert service.image_positions(a.project_id, d['drawing_id'])[0]['equipment_id'] is None
