# Drawing Intelligence V1

## Purpose

Drawing Intelligence adds project-local visual evidence to BuildingAI. It imports a building or HVAC drawing, records detector output, keeps human review separate from the prediction, and lets a reviewer link a confirmed drawing object to an existing BuildingAI equipment record.

## Architecture and workflow

`PyQt page → DrawingService → BaseDrawingDetector → YOLODrawingDetector`

The V1 workflow is **import → detection → bounding box → human review → persistence → manual equipment mapping → read-only Agent query**. Vision is optional and local; no UI page performs model inference directly.

## Current scope

- Supported input formats: PNG, JPG, JPEG. PDF/DWG rendering and topology extraction are outside V1.
- Current legacy model classes: `aircon`, `baseline_mark`, `window`.
- Normalized concepts: `air_conditioning_unit`, `drawing_baseline`, `window`.
- The model has been exercised locally with a real legacy weight; that smoke validates integration, not per-class accuracy.

## Review and safety boundary

Detector output is always stored as `predicted`. A reviewer may mark it `confirmed` or `rejected` and may correct the reviewed class while preserving `original_prediction`. Only a confirmed object can be manually associated with equipment. This is not automatic BEMS-to-drawing matching.

The Agent has only `list_project_drawings`, `get_drawing_detections`, `get_drawing_summary`, and `get_equipment_drawing_location` access. It cannot alter a bounding box, confidence, review state, class, or equipment association. If no confirmed association exists, it returns: `No reliable drawing association has been confirmed for this equipment.`

## Configuration, privacy, and weights

Set `BUILDING_AI_DRAWING_MODEL` or select a local weight on the Drawing Intelligence page. `ultralytics` is an optional dependency: without it or a valid local model, the application starts normally and reports that the model is not configured.

The earlier YOLOv8 research checkout is reference material only, never a BuildingAI runtime dependency. Model weights, drawings, generated inference output, and project databases remain local runtime assets and are excluded from Git. Do not use private drawings in public screenshots or documentation.
