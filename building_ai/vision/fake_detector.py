from __future__ import annotations
from .base import BaseDrawingDetector
from .schemas import DrawingDetection, DrawingModelInfo

class FakeDrawingDetector(BaseDrawingDetector):
    """Deterministic test-only detector; production code never selects it."""
    def __init__(self, detections: list[DrawingDetection] | None = None): self.rows=detections or []
    def is_available(self) -> bool: return True
    def load_model(self) -> None: pass
    def get_model_info(self): return DrawingModelInfo("fake_drawing", "Fake Drawing Detector", "test", "")
    def get_classes(self): return ("aircon", "window", "baseline")
    def detect(self, image): return self.rows
