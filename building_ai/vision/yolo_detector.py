from __future__ import annotations
from pathlib import Path
from .base import BaseDrawingDetector
from .schemas import DrawingDetection, DrawingModelInfo

class YOLODrawingDetector(BaseDrawingDetector):
    def __init__(self, model_info: DrawingModelInfo, confidence: float = 0.25):
        self.info = model_info; self.confidence = confidence; self._model = None
    def is_available(self) -> bool:
        try:
            import ultralytics  # noqa: F401
            return bool(self.info.model_path and Path(self.info.model_path).is_file())
        except ImportError:
            return False
    def load_model(self) -> None:
        if not self.is_available(): raise RuntimeError("Drawing model is not configured or ultralytics is not installed.")
        from ultralytics import YOLO
        self._model = YOLO(self.info.model_path)
    def get_model_info(self) -> DrawingModelInfo: return self.info
    def get_classes(self) -> tuple[str, ...]:
        if self._model is None: return self.info.classes
        names = self._model.names
        return tuple(names.values() if isinstance(names, dict) else names)
    def detect(self, image: str | Path) -> list[DrawingDetection]:
        if self._model is None: self.load_model()
        result = self._model.predict(source=str(image), conf=self.confidence, verbose=False)[0]
        width, height = int(result.orig_shape[1]), int(result.orig_shape[0])
        names = result.names; rows = []
        for box in result.boxes:
            cls = int(box.cls.item()); xyxy = [float(v) for v in box.xyxy[0].tolist()]
            name = names[cls] if isinstance(names, dict) else names[cls]
            rows.append(DrawingDetection(str(name), float(box.conf.item()), *xyxy, width, height))
        return rows
