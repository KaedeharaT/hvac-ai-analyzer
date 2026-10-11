# Historical vision / Dynamo reuse audit

## Scope and evidence standard

This review inspected the user-designated historical research root read-only.
Public references below use relative code names or anonymized asset descriptions;
private drawing identifiers, local absolute paths, source images, models and
coordinate tables are not distributed. No historical script was executed: several
write directly into their input workbooks at import time.

Search proceeded from directory metadata to the small `dynamo_deal` scripts,
selected training/inference metadata, coordinate workbook schemas and values,
and six Dynamo JSON graphs. Environments, vendored libraries and unrelated
research data were excluded from broad content reads. Large weights were first
inspected by file size; one local weight was subsequently loaded through the
current BuildingAI adapter for an explicit inference smoke. This is a scoped
audit, not a claim to have reviewed every file in the historical root.

## Assets and actual execution chain

| Asset | Observed role | Evidence level |
| --- | --- | --- |
| YOLO detection run folders, saved image predictions and 15 label files in one inspected run | Raster drawing inference; five-column normalized YOLO records | Historical output exists; old inference command and exact weight linkage unverified |
| Drawing dataset `data.yaml`, training `args.yaml`, `results.csv`, local `best.pt` / `last.pt` | Three classes: `aircon`, `baseline_mark`, `window`; training image size 640 | Configuration and training artifacts exist; accuracy not independently assessed |
| `sheet_classfy1.py` | Detection CSV grouped by `class_id` into Excel sheets | Code plus related workbook artifacts; referenced original CSV directory is outside the audited root |
| `proportion2.py` | Multiply center columns by `(4 * 7200) / 1632` | Code and numerically consistent processed values; physical calibration unverified |
| `ACCoordinateTrans3.py` | Copy columns 6 and 7 from scaled `class_id_0` into `x,y` | Code plus a 49-object coordinate table; does not itself establish a Revit transform |
| `ACCoordinateDBSCAN4.py` | DBSCAN and Local Outlier Factor; write groups to Excel and plot | Code and two output groups, 21 and 28 rows; clustering is not a room or system assignment |
| `convexhull_and_pillar5.py` | Read intersection coordinates, compute hull, separate interior points | Code and output sheets; comment says input came from Dynamo, but producing graph not located |
| `pillar_is_interior6.py` | Classify pillar positions against equipment groups | Code and classification sheets; geometric assumptions unverified |
| `starrrrt.py` | Intended six-step orchestration | Experimental: imports absent `my_package`; module `main()` functions are placeholders while real work occurs at import time |
| `SQL2EXCEL.py`, `view_the_SQL.py` | SQLite inspection/export | Energy simulation postprocessing; not proof of Revit element creation |
| `Topologic-03` through `Topologic-06b` tutorial graphs | DesignScript geometry / Topologic energy workflows | Six parseable graphs, 26–131 nodes; all inspected graphs have zero saved Revit bindings |
| Modified `Topologic-04-Tutorial02` graph | `TopologicEnergy.EnergyModel.ExportTogbXML` node | Export node exists; connection to the equipment detection pipeline and successful Revit import are unverified |

The recoverable chain is **raster drawing → YOLO detection → bbox / class output
→ CSV-to-sheet grouping → center scaling → equipment coordinate table →
clustering / geometry experiments**. It is not a verified end-to-end
image-to-Revit workflow. No inspected script calls `NewFamilyInstance`, creates
Revit walls/floors/equipment, or supplies a verifiable Revit transaction log.
DesignScript geometry and gbXML export must not be described as Revit import.

## Coordinate findings

Actual bbox corner values and center columns exist. For the inspected aircon
example, averaging the two corners reproduces its pixel center and multiplying
by the fixed coefficient reproduces the scaled center. However:

- Some baseline-marker center values do **not** equal the bbox midpoint.
- An unscaled workbook sheet name does not agree with the contained class ID;
  its scaled counterpart has a different class-to-sheet arrangement.
- Only center columns are scaled; bbox corner columns remain in pixels. The
  output therefore mixes coordinate units without machine-readable unit metadata.
- `7200` appears to be a reference length, `4` a span multiplier, and `1632` a
  pixel span. That interpretation is an inference: no retained calibration record
  proves the measured reference, physical unit, image revision or uncertainty.
- No explicit image-origin-to-building-origin registration, handedness change,
  rotation, crop offset, image-resize history, floor elevation or Revit internal
  unit conversion was recovered.
- There is no verified interactive reference-length/origin selection step.
- The old chain cannot be reproduced safely as-is: it has hard-coded paths,
  import-time writes, missing package structure and inconsistent intermediate
  conventions. Artifact existence proves some processing occurred, not that the
  entire launch script succeeded or that a BIM model is correct.

## Decisions

| Capability | Decision | Reason |
| --- | --- | --- |
| Existing Drawing Viewer + bbox + human review | ADOPT | Already in BuildingAI; reuse its service and vision adapter |
| Confirmed equipment overlay and equipment detail navigation | ADAPT | Keep the existing page; add clickable objects, explicit image-position wording and selection retention |
| Structured position export | ADAPT | Export confirmed **image-relative** bbox centers, not calibrated engineering coordinates |
| Shared 2D building map across drawings | DEFER | Cross-drawing registration, floor identity and units are missing |
| Calibrated coordinate export | DEFER | Scale, origin, orientation and unit provenance are insufficient |
| Optional Revit/BIM interface | DEFER | No verified Revit import chain or reproducible model-creation evidence |
| Copying old orchestration / inferring topology from clusters | REJECT | Unmaintainable side effects and unsupported engineering meaning |

Research value is strongest in an auditable connection between human-confirmed
visual objects and existing BEMS equipment context. The UI links to measured
equipment details and findings; the detector does not assign equipment IDs,
operating states, rooms, floors or piping relationships.

## Implemented boundary and reproducibility

`DrawingService.image_positions` exports only confirmed detections. Records include
`drawing_id`, `object_id`, nullable `equipment_id`, `coordinate_system=image_relative`,
`x`, `y`, `unit=pixel`, top-left origin, right/down axes, bbox, source image SHA-256,
image dimensions, original CV class/confidence, reviewed class, review status and
model ID. `model_version` and `calibration_reference` remain null when unavailable.
`transform_method=bbox_center_no_spatial_calibration` explicitly rules out a physical
transform. Changed source images, dimension mismatches and invalid boxes are rejected.
Predicted/rejected objects are excluded; unassociated confirmed objects keep a null
equipment ID. No local source path is included in the export.

Repeatable synthetic tests: `python -m pytest tests/test_image_positions.py
tests/test_japanese_ui.py`. The synthetic image is 400 × 200 pixels; bbox
`[20,40,80,100]` yields center `(50,70)` pixels, with no engineering-unit claim.
Window zoom/pan does not alter persisted image coordinates. The native GUI QA
script additionally supports actual local inference via explicit `--model` and
`--drawing` options and checks review → manual association → equipment detail →
export. Its association to synthetic BEMS equipment is a software test association,
not evidence that a historical drawing object has that real equipment identity.

An inspected legacy weight was exercised with the current installed Ultralytics
adapter and returned eight real bbox detections. This verifies local adapter
execution only; it does not establish detection accuracy, dataset licensing beyond
the recorded metadata, physical coordinate accuracy or Revit interoperability.

Historical source files were not modified or copied into the product. Revit is
not required to launch or analyze BEMS data. All private inputs, weights and QA
outputs remain outside the tracked source set.
