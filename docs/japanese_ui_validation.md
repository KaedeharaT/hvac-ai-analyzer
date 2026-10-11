# Japanese UI and spatial-position validation

Validated on Windows with the existing PyQt5 desktop application. No Revit
installation is required. Native windows were opened and operated with Qt mouse
input, using the production project, import, semantic review, diagnosis, knowledge,
vision and agent services. The test project is synthetic; it is not customer data.

## Language and scientific invariance

- `zh_CN`, `en_US`, `ja_JP`: 652 keys per locale, identical placeholder names,
  conversions and format specifications. Automated checks reject missing keys,
  empty translations, untranslated English prose and unknown literal UI keys.
- Settings applies the language immediately and persists it in user configuration.
  A separate process reopens the complete persisted project in Japanese.
- Switching languages retains project/equipment/finding selections, energy period
  and chart filters, equipment tabs, knowledge filters, input drafts and chat.
  It neither reruns energy analysis nor reloads the LLM provider.
- Three complete structured energy results are compared for exact equality.
  Existing semantic and diagnosis objects and input frames are also unchanged.
- The UI renderer localizes diagnosis interpretations without rewriting the stored
  scientific interpretation. Source knowledge, raw BEMS names, equipment IDs,
  canonical labels, trace identifiers and numeric evidence remain original.
- General assistant prompts follow the UI language unless the user explicitly
  requests another answer language. Existing deterministic Agent routing and
  research outputs are preserved; this is not a claim of a new Japanese NLU
  benchmark or translation of every stored/model-generated answer.

## Actual desktop exercise

The complete BEMS case contains 96 fifteen-minute records, supply/return water
temperature, flow and power, and one manually associated heat-source equipment.
It is imported through the production import service and reviewed/diagnosed in
actual widgets. Equipment, energy charts, findings, AI Assistant, knowledge search,
drawing review and settings were opened. Closing and restarting a separate app
process retained Japanese and the saved project.

An actual configured local model returned a Japanese HVAC maintenance answer.
Its persisted trace records `operation=generate`, `status=SUCCEEDED`; fallback
text was not counted as model success. An initial cold start exceeded the existing
provider timeout; the model was warmed before the successful UI exercise. No
provider preference or model was changed in the user's configuration.

Actual local YOLO inference returned eight boxes. In the existing Drawing page,
one box was human-confirmed and explicitly associated with the synthetic BEMS
equipment; zoom/pan, equipment navigation and JSON export were exercised.
The association tests software behavior and makes no claim about the real identity
of the historical drawing object. Private weights and drawings are not distributed.
The deterministic 400 x 200 fixture reproduces center (50,70) **pixels**, rejects
unreviewed objects/invalid dimensions, detects source changes, and checks isolation
and provenance. No building-coordinate or Revit claim is made.

## GUI matrix and limits

All three languages were captured across Overview, Data Import, Semantic Mapping,
Equipment, Energy Analysis, Diagnostics, Drawing Intelligence, AI Assistant,
Knowledge Base and Settings, at requested logical sizes 1280x720, 1440x900 and
1920x1080 and Qt scale factors 1.0, 1.25 and 1.5.

Native Windows runs verify the real application at device pixel ratios 1, 1.25
and 1.5. The desktop work area clamps some tall native windows (for example,
1920x1080 becomes 1920x1061 at 100%). Exact requested logical dimensions were
therefore additionally rendered with Qt offscreen at all three scale factors.
The matrix contains 270 native captures and 270 exact-size render captures;
subsequent focused reruns verify layout fixes. Application screenshots were
inspected for CJK glyphs, chart labels, selection contrast and cramped controls.
The OS-wide Windows display setting was not changed: these are application-level
DPI tests, not a claim of three separate Windows display configurations.

Japanese/CJK fonts are registered from installed system fonts. Small drawing
review and diagnostic detail panes scroll to preserve controls and chart axes.
Selected table rows remain readable. Diagnostic detail and chart labels update
when a selected item remains selected across language changes.

## Repeatable checks

```text
python -m pytest
python scripts/run_agentic_evaluation.py
python scripts/run_multi_agent_evaluation.py
python scripts/run_agentic_acceptance.py
python -m pytest tests/test_research_protocol.py tests/test_research_experiment_runner.py tests/test_research_baseline_pipeline.py tests/test_research_adapters.py tests/test_direct_prompt_versions.py
python scripts/qa_trilingual_ui.py --native --scale 1 --workspace artifacts/qa-example
```

Repeat the GUI command with scales `1.25` and `1.5`; omit `--native` for exact-size
rendering. Use a fresh ignored workspace for each run. `--live-llm` exercises an
already configured local provider; `--drawing` and `--model` accept explicitly
supplied local assets. The script writes only under the specified QA workspace
and its isolated user configuration. No credentials, models or private inputs
are bundled with the tests.

Final checks: pytest **183 passed** (four pre-existing third-party deprecation
warnings); single-Agent regression **66 cases, zero failed**; Multi-Agent regression
**66 cases, zero failed**; research subset **20 passed**; Agentic Acceptance
**26 PASS, zero FAIL, zero environment-blocked**. Acceptance is recorded in ignored `artifacts/acceptance/latest.json`; GUI results, screenshots and restart
records stay under ignored `artifacts/trilingual_qa/`.

No GT, frozen splits, semantic algorithms, COP/Delta-T rules, research experiment
code, paper figures or original historical assets were changed. The feature branch
contains source, synthetic tests and documentation only.
