# Historical source-system inventory

> This is an audit of the read-only pre-product source system, not a file map of
> the current repository. Paths under `legacy/` describe migration destinations
> considered during that audit and are not expected to exist in BuildingAI now.

Inventory date: 2026-07-31. Source root:
`<legacy-source-root>` (read-only). The new product root was excluded from all scans.

| Old file | Purpose | Direct imports | Imported by | Migrate? | New destination | Risk |
|---|---|---|---|---|---|---|
| `main.py` | PyQt entry point | `gui_controller`, PyQt5 | direct launch | Reference | `legacy/main.py`; replaced by `app.py` | Low; old shell only |
| `gui_controller.py` | Monolithic Japanese GUI and orchestration | `analysis_core`, `hvac_power_col_memory`, pandas, PyQt5 | `main.py` | Reference, not reused | `legacy/gui_controller.py`; replaced by `building_ai/ui/` | Critical: calls semantic classification independently and mixes UI/domain logic |
| `analysis_core.py` | Import, grouping, weather, COP/load, visualization, Word/Excel export | `evaluation_gt`, research semantic core, LLM client, pandas/numpy/matplotlib/docx/openpyxl | GUI, experiment runner, export test | Selective migration | `legacy/analysis_core.py`; extracted interfaces under `core/`, `services/`, `reports/` | Critical: 119 KB monolith, hard-coded output paths, repeated classification |
| `hvac_power_col_memory.py` | Doctoral semantic method: Direct Prompt, C1–C8, slots, constraints, validation, abstention, fallback, debug/cache | LLM/config/prompts, `experiment.data_matching`, pandas/numpy/matplotlib | analysis core, GUI, experiments, tests | Yes, preserve behavior | `legacy/…` plus formal preserved copy `building_ai/research/…` | Critical research asset; network calls and file side effects remain |
| `llm_client.py` | Ollama/OpenAI/Gemini/Groq/OpenRouter client and compatibility response API | requests, `llm_config` | semantic core, analysis, tests | Reimplemented without secrets | `building_ai/llm/client.py` | Provider API drift; requires integration tests |
| `llm_config.py` | Environment-driven provider/model/prompt selection | environment | LLM client, prompts, experiments | Yes | `building_ai/config/settings.py`, `compat.py` | Low; Direct V1 invariant must remain |
| `direct_prompt_templates.py` | Direct V1–V4 prompt construction and validation | config | semantic core and tests | Yes, preserved | `building_ai/research/direct_prompt_templates.py` | Research reproducibility |
| `evaluation_gt.py` | Ground-truth loading and prediction evaluation | `experiment.eval_utils`, pandas | analysis core, quick evaluator | Reference | `legacy/evaluation_gt.py` | Evaluation should stay separate from human product review |
| `experiment/data_matching.py` | Robust GT/dataframe column-name matching | stdlib | semantic core, experiment scripts, tests | Yes, preserved | `building_ai/research/experiment/data_matching.py` | Merge ambiguity must remain audited |
| `experiment/eval_utils.py` | Canonical labels, merge keys, acceptance policy, metrics | pandas, sklearn | evaluation scripts/tests | Reference pending package migration | `legacy/experiment/eval_utils.py` | Critical for paper metrics, not runtime product logic |
| `experiment/run_experiment.py` | Runs old whole-file analysis | analysis core, research debug helpers | direct execution | Reference | `legacy/experiment/run_experiment.py` | Mutates `sys.path`; unsuitable for product runtime |
| `experiment/run_*experiment.py` | Baseline/full/ablation/direct-prompt experiments | research and evaluation helpers | direct execution | Reference | `legacy/experiment/` | Preserve experimental isolation |
| `test_direct_prompt_versions.py` | Prompt content/version and Full Model invariants | research core/templates/config | pytest/unittest | Migrate essential invariant | `legacy/…`, `tests/test_direct_prompt_versions.py` | Original test assumes top-level modules |
| `test_data_matching.py` | Matching and prompt input tests | matching, research prompt prep | pytest/unittest | Migrate core matcher | `legacy/…`, `tests/test_data_matching.py` | More original cases should be ported |
| `test_eval_merge_keys.py` | Merge-key uniqueness and project identity | experiment eval helper | pytest/unittest | Reference pending | `legacy/…` | Paper-evaluation concern |
| `test_eval_policy.py` | Acceptance/abstention metric policy | experiment eval helper | pytest/unittest | Reference pending | `legacy/…` | Must not conflate UI human review with automated evaluation |
| `test_export.py` | Manual export smoke call | analysis core | direct execution | Reference only | `legacy/test_export.py` | Executes with hard-coded local file; not a unit test |
| `hvac_physical_role_db.json` | Existing role memory/database | read/written by semantic batch function | research core | Reference copy only | `legacy/hvac_physical_role_db.json` | Content provenance/quality unverified; formal runtime does not mutate it |
| `data/*.xlsx`, root CSV/XLSX | Real BEMS datasets | n/a | manual/experiments | Do not copy | remains read-only in old root | Large and potentially sensitive |
| `output/`, `results/`, figures, papers, archives | Experimental artifacts, manuscripts, backups | n/a | reporting/paper workflow | Do not copy | none | Large, duplicated, outside desktop runtime |

## Classification

- Core research algorithm: `hvac_power_col_memory.py`, Direct Prompt templates, matching
  helper, evaluation policy utilities.
- Product/GUI: `main.py`, `gui_controller.py`.
- Mixed runtime algorithm and export: `analysis_core.py`.
- Experimental: `experiment/run_*`, `evaluation_gt.py`, `quick_eval_gt.py`.
- Tests: top-level `test_*.py`; `test_export.py` is a manual script rather than isolated test.
- Temporary/archive: `archive/`, `backup/`, `output/`, old report files and figure scripts.

## Duplicate implementation and coupling findings

1. `gui_controller.py` imports and invokes `batch_physical_role_review`, while
   `analysis_core.analyze_data()` invokes the same batch method and also contains
   `ai_batch_categorize_columns`; results are later merged. This is the primary
   duplicate-semantic-pipeline risk.
2. Unit inference occurs in both the semantic C6 path and `analysis_core`
   (`slot_unit_infer`, `batch_guess_units`, unit-combination scoring).
3. Weather, COP, load, visualization and report export are coupled inside
   `analysis_core.py`.
4. The research batch writes its role database and optionally debug CSV output. Formal
   product persistence must own these side effects instead.
5. Experiment scripts use inconsistent import styles and sometimes modify `sys.path`.
   They remain reference implementations until migrated as a dedicated research package.
