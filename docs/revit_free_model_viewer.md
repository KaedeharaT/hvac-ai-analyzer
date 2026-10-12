# Revit-free model preview: scope, evidence and operation

The optional Model Viewer lives under Drawing Intelligence. It displays existing
OpenStudio surface geometry and explicit equipment references. It is integrated
with the Chinese, English and Japanese UI and the existing confirmed,
image-relative drawing overlays. The renderer remains an optional dependency.

## Historical geometry audit

The user-designated research directory was inspected read-only in stages:
extension/path metadata, selected XML headers, OSM object counts and version
fields, then the vertices of two modest existing models. Environments, vendored
packages and unrelated data were excluded. Original research scripts were not
executed and original files were not edited. Local paths, source models, customer
identifiers, drawings and research workbooks are not distributed here.

| Evidence | Finding | Interpretation |
| --- | --- | --- |
| Small existing OSM example | Version 3.7.0; two spaces, six surfaces; bounds from (0, 0, 0) to (30, 20, 10) m | Actual model vertices, successfully rendered |
| Existing multi-storey tutorial OSM | Version 3.6.1; 25 spaces, 170 surfaces and 48 subsurfaces; bounds from (-10, -10, 0) to (10, 10, 60) m | Actual 218-polygon research model, successfully rendered |
| Later OSM outputs | File and object metadata inspected; additional surface objects exist | Not included in the verified preview cases |
| Tutorial XML files | Measure descriptors; no building geometry in the inspected files | Not usable as gbXML geometry |
| Named gbXML export node in a Dynamo graph | Export code exists; no resulting gbXML file found in the scoped search | Does not establish a usable gbXML model or Revit import |
| glTF, GLB, OBJ, STL, PLY, IFC and VTK file search | No matching model asset found in the scoped search | No claim to have tested these formats |
| Historical detection/coordinate workbooks | Existing pixel/scaled-center data, without a complete calibration record | Remain image-relative; see the [earlier audit](legacy_revit_visualization_audit.md) |

These OSM models provide genuine stored geometry, not evidence that an installed
building or equipment position has been surveyed or verified. Their existence
does not prove successful Revit import or a completed energy simulation.

## Rendering decision

| Candidate | Assessment | Decision |
| --- | --- | --- |
| PyVista + PyVistaQt | Native Qt integration is available, but introduces a separate VTK rendering stack that is not part of this application | Defer |
| Three.js + QtWebEngine | Integrates with the current PyQt5 application; bundled assets work offline; the Qt browser dependency can be optional | Adopt for surface preview |
| Existing 2D Drawing Viewer | Already handles review, equipment overlay and image-relative position export | Keep as the normal drawing workflow |

The rendering package is Three.js 0.186.1, bundled with esbuild 0.28.2. The
application uses its local bundle and Qt's local WebChannel script; there is no
CDN, local HTTP server or runtime npm dependency. The page uses an off-the-record
profile, a restrictive content policy and a request interceptor that permits
only the packaged assets and WebChannel resource. No Revit API, Revit process,
OpenStudio SDK or second 3D renderer is required.

Technical references: [PyVistaQt usage](https://qt.pyvista.org/usage.html),
[PyVista installation](https://docs.pyvista.org/getting-started/installation),
[Qt WebEngine deployment](https://doc.qt.io/archives/qt-5.15/qtwebengine-deploying.html)
and [Three.js OrbitControls](https://threejs.org/docs/pages/OrbitControls.html).

## Supported geometry and coordinate boundary

The adapter deliberately supports **OSM 3.6.1 and 3.7.0** `OS:Surface` and
`OS:SubSurface` polygons whose referenced spaces have zero north rotation and
zero X/Y/Z origins. The corresponding fields and default values were checked
against both versioned OpenStudio IDDs:
[3.6.1 schema](https://github.com/NatLabRockies/OpenStudio/blob/v3.6.1/resources/model/OpenStudio.idd)
and [3.7.0 schema](https://github.com/NatLabRockies/OpenStudio/blob/v3.7.0/resources/model/OpenStudio.idd).

- The coordinate system is `model_local`, the vertex unit is `m`, and Z is up.
  This does not claim survey coordinates or a geographic/true-north alignment.
- Original vertices remain unchanged. Display centering and orbit controls move
  the view; they do not become exported coordinate transforms.
- Nonzero space origins/rotations, other versions, invalid references, repeated
  opening multipliers and invalid polygon geometry are rejected explicitly.
- Limits are 20 MiB per source, 20,000 surface polygons and 128 vertices per
  polygon. These are import bounds, not a performance guarantee at the limits.
- The preview shows surfaces and opening polygons. It does not construct solid
  thickness or cut openings out of walls. Shading/interior partition objects
  are omitted and their counts are recorded when present.
- No pixel-to-model registration, equipment placement, topology inference or
  synthetic floor heights are performed. gbXML, glTF/GLB, OBJ, IFC and RVT are
  not accepted by this adapter. Direct RVT reading is not planned here.

General OSM syntax/conversion, nonidentity space transforms, additional formats
and cross-model registration need their own validated adapters and cases.

## Using the preview

Install the optional Qt dependency in the application's existing environment:

```console
python -m pip install -r requirements-model-viewer.txt
python app.py
```

Open a project, then **Drawing Intelligence → Model Viewer** (`模型预览` /
`モデルビューアー`). Load an existing OSM file that you are authorized to use.
Left-drag rotates, the wheel zooms, right-drag pans, and a click selects a surface.
The object list offers an alternative to clicking. Category checkboxes hide or
show geometry; **Reset view** restores the camera.

Without a model, the viewer displays an empty state. Without QtWebEngine, it
displays an optional-dependency message. BEMS analysis and the 2D drawing
workflow remain usable in both cases. The standard application entry point
prepares Qt5's optional shared OpenGL context before constructing QApplication.

Object names and IDs are original source data. Controls, categories, empty
states, guidance and KPI labels switch immediately with the existing Chinese,
English and Japanese LanguageManager. Switching language neither changes the
model coordinates nor recomputes energy or diagnosis results.

## Equipment references and scientific provenance

An unassociated object says **No equipment association established**. To create
a reference, choose an equipment item from the current project and supply a
reviewer and evidence note. The service resolves membership using the existing
ProjectStore semantic records and EquipmentService. It does not infer an
equipment identity from a surface name or geometry.

The association is explicitly `reference_only_not_equipment_position`. It means
that a person has identified a reference object, not that equipment occupies a
surface vertex or its centroid. The details panel displays existing equipment
signals, mean COP, mean ΔT, deterministic findings and any already confirmed
drawing references; the equipment button opens the existing detail page.
Absent measurements remain unavailable. No analysis rule is added or changed.

Imports create project-scoped snapshots in the normal runtime project directory.
The SHA-256 of the actual source bytes is checked on load. Model snapshots,
manifests and human association history are separate from original engineering
inputs and from paper/GT/experiment artifacts. Re-importing identical source
bytes preserves the association history.

Units, geometry and engineering-validation status are derived from the checked
source adapter; manifest fields cannot override them. Review history must match
the project, model hash, object and coordinate metadata. A failed reload clears
the previous scene and disables association/export controls, so a stale model
cannot be presented under a different selected source. Closing the dialog
releases its renderer.

**Export provenance** writes a JSON record with source name, source/model hash,
source format/version, adapter version, coordinate system, units, transform
method, engineering-validation boundary, object identities, omissions and
timestamped human review events. It does not export inferred equipment XYZ.
`drawing_hash` is explicitly null because no drawing-to-model registration has
been established; another drawing linked to the same equipment does not prove
such registration. Original absolute paths are omitted. Users should treat
their model names, review notes and exported records as project data.

## Verification

Automated tests cover source vertices/units/hashes, invalid geometry and
transforms, subsurface references, explicit reviews, project isolation, tamper
detection, re-import/restart, provenance export and the application with the
optional renderer unavailable. Trilingual tests compare all keys/placeholders
and exact structured energy outputs, and ensure locale switching does not
change diagnostics, raw data, provider or current analysis objects.

```console
python -m pytest
python scripts/run_agentic_acceptance.py
python scripts/run_multi_agent_evaluation.py
python scripts/qa_model_viewer.py --model "<authorized-existing-model.osm>" --workspace artifacts/model-viewer-qa --scale 1.25
```

The native Windows Qt/WebGL walkthrough rendered both existing models. It
performed real mouse rotation, zoom, pan and raycast selection, category
filtering, a manual reference to a **synthetic BEMS equipment item**, equipment
navigation, provenance export and a separate-process reopen. That test
association is software QA, not historical installation evidence. It also
checked the unassociated and no-model states. No Revit code was loaded.

Three UI languages and requested window sizes 1280×720, 1440×900 and 1920×1080
were captured at Qt scale factors 1.0, 1.25 and 1.5. The desktop work area clamps
some requested heights; actual sizes/DPR are recorded in the local QA report.
This verifies application scaling on the available Windows desktop, not changes
to the machine-wide Windows DPI setting or a guarantee for every GPU driver.
Private model copies, screenshots and reports stay in ignored local artifacts.

To rebuild the offline renderer from pinned dependencies:

```console
npm ci --prefix tools/model_viewer --ignore-scripts
npm run --prefix tools/model_viewer build
```

The bundled Three.js MIT and Earcut ISC licenses are shipped next to the assets;
see the [third-party inventory](model_viewer_third_party.md) for optional Qt
dependency terms. The original
research assets, model weights, BEMS data, databases and temporary files are
excluded from Git. No release tag is created or moved for this work.

The public README screenshot is generated with
`python scripts/capture_model_preview.py`. This helper creates a fictional
33-surface OSM input and synthetic BEMS, exercises the real Qt/WebGL viewer,
and copies only its native screenshot into `docs/images/model-viewer.png`.
It accepts no historical input model. Generated OSM, BEMS and runtime records
remain under ignored artifacts; the screenshot demonstrates software behavior,
not a reconstructed or verified building.
