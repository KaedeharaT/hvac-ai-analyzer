from __future__ import annotations
import hashlib, shutil, uuid
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
        with self.database.connect() as c: c.execute("INSERT INTO drawings VALUES (:drawing_id,:project_id,:file_name,:file_type,:page_count,:managed_path,:model_id,:status,:imported_at,'{}')", row)
        return row
    def list_drawings(self, project_id: str) -> list[dict]:
        with self.database.connect() as c: return [dict(r) for r in c.execute("SELECT * FROM drawings WHERE project_id=? ORDER BY imported_at DESC", (project_id,))]
    def get_drawing(self, project_id: str, drawing_id: str) -> dict:
        with self.database.connect() as c: r=c.execute("SELECT * FROM drawings WHERE project_id=? AND drawing_id=?",(project_id,drawing_id)).fetchone()
        if not r: raise KeyError(drawing_id)
        return dict(r)
    def save_detections(self, project_id: str, drawing_id: str, model_id: str, detections: list[DrawingDetection]) -> list[dict]:
        drawing=self.get_drawing(project_id,drawing_id); now=self._now(); rows=[]
        with self.database.connect() as c:
            c.execute("DELETE FROM drawing_detections WHERE project_id=? AND drawing_id=?",(project_id,drawing_id))
            for item in detections:
                row={"detection_id":str(uuid.uuid4()),"project_id":project_id,"drawing_id":drawing_id,"page_number":item.page_number,"class_name":item.class_name,"normalized_class":item.normalized_class,"original_prediction":item.class_name,"reviewed_class":None,"confidence":item.confidence,"bbox_x1":item.bbox_x1,"bbox_y1":item.bbox_y1,"bbox_x2":item.bbox_x2,"bbox_y2":item.bbox_y2,"image_width":item.image_width,"image_height":item.image_height,"model_id":model_id,"review_status":"predicted","equipment_id":None,"created_at":now}; rows.append(row)
                c.execute("""INSERT INTO drawing_detections VALUES (:detection_id,:project_id,:drawing_id,:page_number,:class_name,:normalized_class,:original_prediction,:reviewed_class,:confidence,:bbox_x1,:bbox_y1,:bbox_x2,:bbox_y2,:image_width,:image_height,:model_id,:review_status,:equipment_id,:created_at)""",row)
            c.execute("UPDATE drawings SET model_id=?,status='detected' WHERE drawing_id=?",(model_id,drawing_id))
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
    def map_equipment(self, project_id: str, detection_id: str, equipment_id: str) -> None:
        with self.database.connect() as c:
            row=c.execute("SELECT review_status FROM drawing_detections WHERE project_id=? AND detection_id=?",(project_id,detection_id)).fetchone()
            if not row: raise KeyError(detection_id)
            if row['review_status']!='confirmed': raise ValueError('Only confirmed detections can be mapped to equipment.')
            c.execute("UPDATE drawing_detections SET equipment_id=? WHERE detection_id=?",(equipment_id,detection_id))
    def equipment_location(self, project_id: str, equipment_id: str) -> list[dict]:
        with self.database.connect() as c:return [dict(r) for r in c.execute("SELECT d.file_name,d.drawing_id,x.* FROM drawing_detections x JOIN drawings d ON x.drawing_id=d.drawing_id WHERE x.project_id=? AND x.equipment_id=? AND x.review_status='confirmed'",(project_id,equipment_id))]
    def summary(self, project_id: str) -> dict:
        rows=self.list_detections(project_id); counts={}
        for r in rows: counts[r['review_status']]=counts.get(r['review_status'],0)+1
        return {"drawings":len(self.list_drawings(project_id)),"detections":len(rows),"review_status_counts":counts}
