# Native desktop functional validation

Validation date: 2026-10-12. This is software acceptance evidence, separate from
doctoral experiment results, ground truth, frozen splits and paper figures.

## Environment and evidence

The application was launched through `python app.py` on an interactive Windows
desktop. Subsequent automation used the real production Qt windows, file/input
dialogs, buttons, review forms and WebGL mouse events. These were native Windows
operations, not offscreen rendering or mocked import/inference/LLM results.

Tests used separate configuration directories, SQLite databases and managed
copies of inputs. Three pre-existing user projects were preserved. Original
workbook hashes were rechecked. Private assets, model weights, database files,
screenshots containing source data and detailed manifests remain local under
ignored `artifacts/full_functional_qa/`.

A local `.venv` meets the repository dependency constraints, including pandas
2.3.3, openpyxl 3.1.5 and pyarrow 19.0.1. It inherits the existing optional Qt,
PyTorch and Ultralytics installation; the original research environment was not
upgraded. Core workflows were also exercised with the original pandas 1.5.3
environment. Inherited Conda-only dependency conflicts are outside the desktop
application's dependency set.

## Actual source projects

Names and paths are withheld from this public document. Each input had its own
project identity and managed copy. No filename was treated as a date or as an
equipment association.

| Source | Observed structure | Native desktop result and limits |
| --- | --- | --- |
| A | 1,440 readings, 47 columns; Japanese headers; clock only | Import, semantic review and diagnosis completed. Calendar-based charts correctly unavailable without a source date. |
| B | 1,440 readings, 59 columns; two preceding measurement/unit rows | Unit rows excluded from readings and recorded as evidence. Power/energy and temperatures available. Two source-labelled temperature points reviewed against their headers; device COP lacks flow and device power. |
| C | 44,640 readings, 57 columns; mixed meter and temperature signals | Large import completed with event processing. Energy/temperature display and diagnosis completed. Nonmonotonic energy readings carry an interval-value warning; meter accumulation conventions require source verification. |
| D | 40,320 readings, 119 columns; separate unnamed date and clock | Actual minute timestamps restored from the two columns. Temperature charts available; missing equipment inputs are reported rather than invented. |

This verifies that different real research exports can be loaded and inspected.
It does not certify their calibration, physical equipment identity or accuracy.
The requested additional source called Project7 was not found by that filename.

Nine valid synthetic datasets and two deliberately malformed files covered UTF-8,
CP932, GB18030, UTF-16, CSV delimiters/preambles, Excel worksheet selection, XLS,
Chinese/English/Japanese/mixed labels, W/kW and L/s/m³/h, missing units/signals,
missing/duplicate times, missing readings, reordered columns, duplicate headers
and Unicode normalization collisions. Malformed imports displayed handled
errors without replacing the current project data. Known complete synthetic
signals produced COP and ΔT using the existing deterministic services.

## Drawing and model execution

Two distinct historical drawing styles were imported through the GUI. The real
historical three-class detector produced 8 and 4 predictions respectively.
Its vocabulary is `aircon`, `baseline_mark`, `window`. Confirm/reject, explicitly
manual equipment references, zoom/pan, detail navigation and image-position
JSON exports were exercised. This demonstrates inference execution, not detector
accuracy or verified physical identity.

The preferred older checkpoint requires an unavailable custom `RIA` layer. A
compatible existing historical checkpoint was validated and selected instead;
no generic pretrained detector was presented as the custom drawing model.
Weights and their private local source/hash record are not published.
The compatible model was selected through the original application's file dialog
and persisted in user settings. Other settings and the three existing projects
were checked unchanged.

Existing OpenStudio OSM 3.7.0 (6 surfaces) and 3.6.1 (218 surfaces/openings) files
were actually rendered. Rotation, zoom, pan, surface selection, category toggles,
reset, no-association notice, explicit equipment-reference review and export
worked. Model-local coordinates remain metres. These historical research/example
geometries are not certified as-built BIM. Software-test references to synthetic
BEMS were explicitly labelled as unrelated QA references, not installation
evidence. Neither Revit nor RVT import was used.
Two additional empty project identities verified drawing-only and model-only
loading without BEMS or equipment. Both assets reopened in a separate process
without creating an equipment association.

## AI, persistence and language

The configured local `qwen2.5:7b` service completed actual Japanese generation.
Project questions used real read-only tools for equipment, energy, COP, ΔT,
diagnostics, suggestions and unavailable rated capacity. Native project switching
verified that an absent equipment ID did not return another project's metrics.
Knowledge Base search returned attributed local catalog results.

Separate application processes reopened both the real-project database and the
combined synthetic project. Project data, human label/unit/ownership changes,
drawing reviews, image-position exports, model geometry, association events and
Japanese preference were restored. All main pages were opened in Chinese,
English and Japanese at multiple window sizes. Language switching preserved the
canonical numerical summary.
Native captures covered 1280×720, 1440×900 and 1920×1080 requests with Qt scale
factors 1, 1.25 and 1.5 using Yu Gothic. These scale-factor checks did not change
the Windows system DPI setting; physical-screen bounds can limit visible window
area at the largest scaled size. The source screenshots retain actual sizes.

## Repairs

Thirteen grouped reliability issues were addressed:

1. New-project and rename actions retained another project's live context;
   active jobs also needed protection against input replacement and premature close.
2. Clock-only imports acquired today's date, and unnamed date/clock pairs lost
   minute resolution.
3. Logger metadata/unit rows and common encoded/preamble CSV layouts were not
   preserved as import evidence.
4. Import exceptions escaped Qt slots and large workbook reads blocked the UI.
5. Unicode-normalized duplicate headers collided during semantic processing.
6. Assumed units could be used without source evidence; unit review, confirmed
   abstentions and re-running semantics needed consistent persistence.
7. Valid non-heat-source energy rankings failed heat-source-only consistency checks.
8. Missing timestamp/calculation reasons were too generic, with misleading
   default date controls and zero-valued unavailable KPI summaries.
9. A usable drawing model was not configured in the user environment.
10. The detection list changed ordering after the first review.
11. Agent focus overrode explicit equipment requests, and specific energy or
    rated-load questions were answered with a generic KPI summary.
12. Semantic review fields could retain another project's selected equipment;
    changing project now clears the form while language changes preserve edits.
13. Closed windows could receive global language changes after Qt had destroyed
    child widgets. Connections now stop on close and resume when shown again.

Research classifiers, COP/ΔT equations, diagnosis rules and frozen evaluation
definitions were not changed. Product input validation and human review were
corrected. Consequently, previously misread timestamps or unsupported assumed
units can intentionally produce different availability from the faulty import.

## Final checks

| Check | Result |
| --- | --- |
| Full pytest | 209 passed (4 third-party deprecation warnings) |
| Single-Agent deterministic regression | 66/66 |
| Multi-Agent deterministic regression | 66/66 |
| Agentic acceptance | 26 passed, 0 failed, 0 environment-blocked |
| Dedicated research tests | 19 passed; also included in full pytest |
| Three-language coverage, placeholders and numerical invariance | Passed in full pytest and native language switching |
| Final native GUI smoke | 90 captures, no unhandled exceptions; restart passed |

The configured local LLM was exercised during this session; the separate existing
52-case E2E artifact checked by the acceptance runner was not regenerated by this
desktop exercise. No remote CI run or main-branch merge is claimed for this fix
branch. Private-data calibration, missing source dates/units and incomplete COP
inputs remain data-review requirements, not successful calculated results.

## Reproduction

Use a new ignored workspace. Supply a local JSON manifest containing `bems`
entries with `name` and `path`, optional `sheet` and review instructions. The
manifest may also specify local `weights`, `drawings`, `model`, and `questions`.
Do not publish that manifest when it references private assets.

```powershell
python scripts/create_functional_fixtures.py artifacts/functional-inputs
python scripts/qa_functional_desktop.py --manifest <local-manifest.json> --workspace artifacts/functional-run
python scripts/qa_functional_desktop.py --manifest <local-manifest.json> --workspace artifacts/functional-run --reopen
python scripts/qa_functional_desktop.py --manifest <local-manifest.json> --workspace artifacts/assets-run --assets-only
python scripts/qa_functional_desktop.py --manifest <local-manifest.json> --workspace artifacts/assets-run --assets-only --reopen
python -m pytest
python scripts/run_agentic_evaluation.py
python scripts/run_multi_agent_evaluation.py
python scripts/run_agentic_acceptance.py
```

Synthetic XLS generation additionally uses optional `xlwt`; importing XLS uses
`xlrd`. All generated fixtures are software tests, not research observations.
Detailed machine-readable results and native screenshots remain in the ignored
local evidence directory. No private raw source, weight or database is required
by the committed regression tests.
