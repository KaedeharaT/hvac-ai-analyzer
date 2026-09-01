# Historical migration report

> This document preserves the initial research-to-product migration record. It
> is not a current capability or validation report. References to `legacy/`
> describe the migration target considered at that time; they do not claim that
> the current repository contains a live `legacy/` directory. For current
> behavior, use the README, architecture documentation, and CI.

## Safety boundary

The old root was inspected and copied only. No old file was edited, formatted, moved,
renamed, or deleted. Large datasets, model artifacts, outputs, papers, caches, Git data,
and images were not copied.

At final verification, SHA-256 hashes matched for all 14 top-level files copied into
`legacy/` (14/14). The old repository already contained many staged/modified entries
before this migration; they were left untouched.

## Direct reference copies

`legacy/` contains the old entry point, GUI, analysis monolith, semantic research core,
LLM/config/prompts, ground-truth helper, role database, key tests, and core experiment
helpers/runners. It is a reference implementation and is not imported by formal code.

## Formal migration

- LLM configuration became a typed `Settings` object. Providers remain Ollama, OpenAI,
  Gemini, Groq and OpenRouter. Defaults remain Ollama/local open-source LLM/local port 11434.
- A provider-neutral LLM client keeps temperature 0 and seed 0 defaults. No keys are
  embedded.
- The research semantic source and prompt/matching dependencies were copied into
  `building_ai/research/` and imports were package-qualified. C1–C8 logic was not
  rewritten. Full Model still explicitly calls `direct_v1_name`.
- The product gets one semantic entry: `SemanticService.analyze_dataframe()`.
  It returns one `AnalysisResult` reused by storage, equipment, analytics, UI and Agent.
- A conservative offline backend was added. This is a product fallback, not a claimed
  equivalent of the research model. It abstains when evidence is insufficient.
- Project/domain models, SQLite project/semantic/review persistence, Parquet time-series
  persistence, import metadata, and separate AI/human labels were added.
- V1.1 replaces `(project_id, raw_name)` identity with deterministic UUIDv5 `point_id`
  over project/source/sheet/raw-name provenance. SQLite schema version 2 migrates existing
  semantic and review rows explicitly instead of silently recreating the database.
- Heat-source COP and load-ratio interfaces were extracted. They preserve the standard
  water-side formula and broad legacy valid COP range (0.5–15), but do not yet reproduce
  every legacy grouping, sampling and unit-combination branch.
- The old monolithic GUI was replaced by an English PyQt navigation shell and eight pages.
- Agent V1 uses a plain tool registry. It does not receive a whole DataFrame and rejects
  unverified ABSTAIN points.
- The research backend now receives `ResearchLLMClientAdapter`, which implements the
  `client.chat.completions.create(...)` response shape expected by preserved helpers.
- Research units now come from cached C6 output. Product header-unit inference is confined
  to offline product mode.

## Deliberate behavior changes

1. Formal storage paths are relative to the new project settings, not the old absolute
   machine-specific legacy output paths.
2. The app defaults to conservative offline semantics so it works without Ollama. Users
   or experiments must deliberately select the research backend.
3. Human corrections are stored in a separate table; the AI prediction is immutable.
4. Weather failure returns `None` and cannot crash local semantic work.
5. No BAS control or protocol write API exists.
6. Generic `Power/電力`, `Flow/流量`, and `Capacity` without equipment evidence are stored
   as physical-quantity evidence but receive `other` plus REVIEW/ABSTAIN, not an assumed
   heat-source label.

## Partially migrated / deferred

- Terminal COP, complete legacy grouping, cumulative-energy power derivation, weather
  correlation, charts, and exports need behavior-by-behavior extraction.
- The UI worker abstraction exists, but every import/semantic call is not yet wired to
  a background thread with progress and cancellation.
- Equipment extraction is heuristic. There is no claimed ontology or knowledge graph.
- Original evaluation-policy and merge-key suites remain in legacy pending a clean
  research-test package migration.

## Testing and known issues

See `docs/regression_report.md`. Research-backend end-to-end testing requires a reachable
LLM and a controlled fixture. Real BEMS regression remains pending; no numerical equivalence
is claimed without that comparison.

V1 foundation validation: 12 product tests passed, all formal Python modules compiled,
the preserved research module imported, an offscreen eight-page GUI smoke test passed,
and `python app.py` remained running until the smoke-test process was deliberately stopped.

V1.1 stabilization validation is recorded in `docs/regression_report.md`. Before applying
schema v2 to the local development database, a recoverable copy was placed at
`temp/building_ai_pre_v2.sqlite3.bak`.
