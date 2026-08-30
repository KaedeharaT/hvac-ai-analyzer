from __future__ import annotations
import os
from pathlib import Path
from .schemas import DrawingModelInfo

class DrawingModelRegistry:
    """Settings/environment backed registry; no research-machine paths are embedded."""
    ENV_KEY = "BUILDING_AI_DRAWING_MODEL"
    def __init__(self, model_path: str = ""):
        self.model_path = model_path or os.getenv(self.ENV_KEY, "")
    def configured(self) -> bool:
        return bool(self.model_path and Path(self.model_path).is_file())
    def get(self) -> DrawingModelInfo:
        return DrawingModelInfo("legacy_yolov8_drawing", "Legacy YOLOv8 Drawing Detector", "ultralytics", self.model_path, ("aircon", "window", "baseline"), "legacy", "Detects only the legacy research classes: Air Conditioner, Window, and Baseline.")
