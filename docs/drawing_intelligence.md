# Drawing Intelligence V1

BuildingAI Drawing Intelligence is derived from earlier research on YOLOv8-based architectural drawing object detection, but is reimplemented as a project-local service and detector adapter. It never imports research code or depends on a research checkout at runtime.

V1 imports PNG/JPG/JPEG drawings, stores managed copies per project, runs an optional Ultralytics detector, persists its raw bounding boxes and confidence, and separates `predicted`, `confirmed`, and `rejected` review states. Only confirmed detections may be manually associated with BuildingAI equipment. The Agent exposes read-only drawing queries and will abstain when no confirmed association exists.

The legacy research model supports only `aircon`, `window`, and `baseline` / `baseline_mark`, normalized respectively to `air_conditioning_unit`, `window`, and `drawing_baseline`. Other HVAC classes are future extensions, not current model claims.

Set `BUILDING_AI_DRAWING_MODEL` or configure the local model path in application settings. `ultralytics` is an optional local dependency: without it or a valid model file, the application continues normally and the Drawing Intelligence page reports that the model is not configured. User drawings and model weights are runtime data and must not be committed.
