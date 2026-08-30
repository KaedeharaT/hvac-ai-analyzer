from __future__ import annotations
from dataclasses import dataclass, field

NORMALIZED_CLASSES = {"aircon": "air_conditioning_unit", "window": "window", "baseline": "drawing_baseline", "baseline_mark": "drawing_baseline"}

@dataclass(frozen=True)
class DrawingModelInfo:
    model_id: str
    display_name: str
    framework: str
    model_path: str
    classes: tuple[str, ...] = ()
    model_version: str = ""
    description: str = ""

@dataclass(frozen=True)
class DrawingDetection:
    class_name: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    image_width: int
    image_height: int
    normalized_class: str = ""
    page_number: int = 1

    def __post_init__(self):
        object.__setattr__(self, "normalized_class", self.normalized_class or NORMALIZED_CLASSES.get(self.class_name.casefold(), self.class_name.casefold()))

    def to_dict(self) -> dict:
        return self.__dict__.copy()
