from __future__ import annotations
import hashlib, shutil, uuid, json, math
from datetime import datetime, timezone
from pathlib import Path
from building_ai.vision.schemas import DrawingDetection

ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

class DrawingService:
    def __init__(self, database, data_root: str | Path): self.database = database; self.root = Path(data_root)
    @staticmethod
    def _now(): return datetime.now(timezone.utc).isoformat()
    def import_drawing(self, project_id: str, source: str | Path) -> dict:
        source = Path(source)
        if source.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES: raise ValueError("Drawing V1 supports PNG, JPG, and JPEG images.")
        digest = hashlib.sha256(source.read_bytes()).hexdigest(); target = self.root / project_id / "drawings" / f"{digest[:12]}_{source.name}"; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        row = {"drawing_id": str(uuid.uuid4()), "project_id": project_id, "file_name": source.name, "file_type": source.suffix.lower().lstrip("."), "page_count": 1, "managed_path": str(target), "model_id": None, "status": "imported", "imported_at": self._now()}
        row['payload_json'] = json.dumps({'source_image_hash': digest})
        with self.database.connect() as c: c.execute("INSERT INTO drawings VALUES (:drawing_id,:project_id,:file_name,:file_type,:page_count,:managed_path,:model_id,:status,:imported_at,:payload_json)", row)
        return row
    def list_drawings(self, project_id: str) -> list[dict]:
        with self.database.connect() as c: return [dict(r) for r in c.execute("SELECT * FROM drawings WHERE project_id=? ORDER BY imported_at DESC", (project_id,))]
    def get_drawing(self, project_id: str, drawing_id: str) -> dict:
        with self.database.connect() as c: r=c.execute("SELECT * FROM drawings WHERE project_id=? AND drawing_id=?",(project_id,drawing_id)).fetchone()
        if not r: raise KeyError(drawing_id)
        return dict(r)
    def save_detections(self, project_id: str, drawing_id: str, model_id: str, detections: list[DrawingDetection], *, model_provenance: dict | None = None) -> list[dict]:
        drawing=self.get_drawing(project_id,drawing_id); now=self._now(); rows=[]
        with self.database.connect() as c:
            c.execute("DELETE FROM drawing_detections WHERE project_id=? AND drawing_id=?",(project_id,drawing_id))
            for item in detections:
                row={"detection_id":str(uuid.uuid4()),"project_id":project_id,"drawing_id":drawing_id,"page_number":item.page_number,"class_name":item.class_name,"normalized_class":item.normalized_class,"original_prediction":item.class_name,"reviewed_class":None,"confidence":item.confidence,"bbox_x1":item.bbox_x1,"bbox_y1":item.bbox_y1,"bbox_x2":item.bbox_x2,"bbox_y2":item.bbox_y2,"image_width":item.image_width,"image_height":item.image_height,"model_id":model_id,"review_status":"predicted","equipment_id":None,"created_at":now}; rows.append(row)
                c.execute("""INSERT INTO drawing_detections VALUES (:detection_id,:project_id,:drawing_id,:page_number,:class_name,:normalized_class,:original_prediction,:reviewed_class,:confidence,:bbox_x1,:bbox_y1,:bbox_x2,:bbox_y2,:image_width,:image_height,:model_id,:review_status,:equipment_id,:created_at)""",row)
            c.execute("UPDATE drawings SET model_id=?,status='detected' WHERE drawing_id=?",(model_id,drawing_id))
            payload=json.loads(drawing.get('payload_json') or '{}')
            payload['model_provenance']=model_provenance
            payload['associations']={}
            c.execute("UPDATE drawings SET payload_json=? WHERE project_id=? AND drawing_id=?",
                      (json.dumps(payload,ensure_ascii=False),project_id,drawing_id))
        return rows
    def list_detections(self, project_id: str, drawing_id: str | None = None, *, confirmed_only=False) -> list[dict]:
        sql="SELECT * FROM drawing_detections WHERE project_id=?"; args=[project_id]
        if drawing_id: sql+=" AND drawing_id=?"; args.append(drawing_id)
        if confirmed_only: sql+=" AND review_status='confirmed'"
        sql+=" ORDER BY created_at,detection_id"
        with self.database.connect() as c:return [dict(r) for r in c.execute(sql,args)]
    def review_detection(self, project_id: str, detection_id: str, status: str, reviewed_class: str | None=None) -> None:
        if status not in {'confirmed','rejected'}: raise ValueError(status)
        with self.database.connect() as c:
            exists=c.execute("SELECT 1 FROM drawing_detections WHERE project_id=? AND detection_id=?",(project_id,detection_id)).fetchone()
            if not exists: raise KeyError(detection_id)
            c.execute("UPDATE drawing_detections SET review_status=?, reviewed_class=COALESCE(?,reviewed_class) WHERE detection_id=?",(status,reviewed_class,detection_id))
            row=c.execute("SELECT drawing_id FROM drawing_detections WHERE detection_id=?",(detection_id,)).fetchone()
            drawing=c.execute("SELECT payload_json FROM drawings WHERE drawing_id=?",(row['drawing_id'],)).fetchone()
            payload=json.loads(drawing['payload_json'] or '{}')
            payload.setdefault('object_reviews',[]).append({'object_id':detection_id,'status':status,'reviewed_class':reviewed_class,'reviewed_at':self._now()})
            # Re-review invalidates previous identity evidence until re-associated.
            payload.setdefault('associations',{}).pop(detection_id,None)
            c.execute("UPDATE drawings SET payload_json=? WHERE drawing_id=?",(json.dumps(payload,ensure_ascii=False),row['drawing_id']))
    def map_equipment(self, project_id: str, detection_id: str, equipment_id: str, *,
                      association_status: str = 'pending', reviewer: str = '', evidence_note: str = '') -> None:
        if association_status not in {'pending', 'verified', 'test'}:
            raise ValueError('Unsupported association status.')
        if association_status == 'verified' and (not reviewer.strip() or not evidence_note.strip()):
            raise ValueError('Verified association requires a reviewer and source evidence.')
        if association_status == 'verified':
            from building_ai.storage import ProjectStore
            from .equipment_service import EquipmentService
            organization = EquipmentService().organize(project_id, ProjectStore(self.database).load_semantics(project_id))
            if equipment_id not in {item.equipment_id for item in organization.equipment}:
                raise ValueError('Verified association requires equipment in the current project.')
        with self.database.connect() as c:
            row=c.execute("SELECT review_status,drawing_id FROM drawing_detections WHERE project_id=? AND detection_id=?",(project_id,detection_id)).fetchone()
            if not row: raise KeyError(detection_id)
            if row['review_status']!='confirmed': raise ValueError('Only confirmed detections can be mapped to equipment.')
            owner=c.execute("SELECT project_id FROM equipment WHERE equipment_id=?",(equipment_id,)).fetchone()
            if owner and owner['project_id'] != project_id:
                raise ValueError('Equipment belongs to another project.')
            c.execute("UPDATE drawing_detections SET equipment_id=? WHERE detection_id=?",(equipment_id,detection_id))
            drawing=c.execute("SELECT payload_json FROM drawings WHERE project_id=? AND drawing_id=?",(project_id,row['drawing_id'])).fetchone()
            payload=json.loads(drawing['payload_json'] or '{}')
            event={'object_id':detection_id,'equipment_id':equipment_id,'association_status':association_status,
                   'reviewer':reviewer.strip(),'evidence_note':evidence_note.strip(),'reviewed_at':self._now()}
            payload.setdefault('association_events',[]).append(event)
            payload.setdefault('associations',{})[detection_id]=event
            c.execute("UPDATE drawings SET payload_json=? WHERE project_id=? AND drawing_id=?",
                      (json.dumps(payload,ensure_ascii=False),project_id,row['drawing_id']))

    def association_record(self, project_id: str, drawing_id: str, detection_id: str) -> dict:
        payload=json.loads(self.get_drawing(project_id,drawing_id).get('payload_json') or '{}')
        return payload.get('associations',{}).get(detection_id, {'association_status':'pending'})
    def equipment_location(self, project_id: str, equipment_id: str) -> list[dict]:
        with self.database.connect() as c:return [dict(r) for r in c.execute("SELECT d.file_name,d.drawing_id,x.* FROM drawing_detections x JOIN drawings d ON x.drawing_id=d.drawing_id WHERE x.project_id=? AND x.equipment_id=? AND x.review_status='confirmed'",(project_id,equipment_id))]
    def summary(self, project_id: str) -> dict:
        rows=self.list_detections(project_id); counts={}
        for r in rows: counts[r['review_status']]=counts.get(r['review_status'],0)+1
        return {"drawings":len(self.list_drawings(project_id)),"detections":len(rows),"review_status_counts":counts}

    def image_positions(self, project_id: str, drawing_id: str) -> list[dict]:
        """Confirmed image-space centers only; never an engineering transform.

        Unassociated objects retain a null equipment ID. Rejected/predicted
        objects are excluded, and no unit, origin or calibration is inferred.
        """
        from PIL import Image
        drawing = self.get_drawing(project_id, drawing_id)
        source = Path(drawing['managed_path'])
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        original = json.loads(drawing.get('payload_json') or '{}').get('source_image_hash')
        if original and original != digest:
            raise ValueError('Source image changed after import; re-import and review it.')
        with Image.open(source) as image:
            width, height = image.size
        records = []
        for row in self.list_detections(project_id, drawing_id, confirmed_only=True):
            x1, y1, x2, y2 = (row[key] for key in ('bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'))
            if ((width, height) != (row['image_width'], row['image_height'])
                    or not all(math.isfinite(v) for v in (x1, y1, x2, y2))
                    or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height)):
                raise ValueError('Bounding box does not match the original image dimensions.')
            records.append({
                'drawing_id': drawing_id, 'object_id': row['detection_id'],
                'equipment_id': row['equipment_id'], 'coordinate_system': 'image_relative',
                'x': (x1 + x2) / 2, 'y': (y1 + y2) / 2, 'unit': 'pixel',
                'origin': 'top_left', 'x_direction': 'right', 'y_direction': 'down',
                'transform_method': 'bbox_center_no_spatial_calibration',
                'calibration_reference': None, 'review_status': row['review_status'],
                'model_id': row['model_id'], 'model_version': None,
                'source_image_hash': digest, 'image_width': width, 'image_height': height,
                'bbox_xyxy': [x1, y1, x2, y2],
                'cv_prediction': row['original_prediction'], 'confidence': row['confidence'],
                'human_class': row['reviewed_class'],
                'association_review': self.association_record(project_id,drawing_id,row['detection_id']),
                'model_provenance': json.loads(drawing.get('payload_json') or '{}').get('model_provenance'),
            })
        return records

    def export_image_positions(self, project_id: str, drawing_id: str, destination: str | Path) -> None:
        records = self.image_positions(project_id, drawing_id)
        Path(destination).write_text(json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
