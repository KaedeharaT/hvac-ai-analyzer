# BuildingAI research-readiness audit

## Scope and standard

This audit treats BuildingAI as a doctoral-research platform rather than a
product dashboard. A thesis result is acceptable only when its dataset hash,
data revision, mapping, configuration, code state and generated artifacts can
be inspected independently of the UI.

The new UI-independent research runner is:

```powershell
python scripts/run_research_experiment.py --config research_config.json
```

It writes an ignored, immutable directory under `artifacts/experiments/` with
an experiment ID, input SHA-256, data-quality metadata, semantic/equipment
mappings, KPI time series, diagnostics, full-precision summaries, config,
Git/environment snapshots, a hash manifest, and a replay command. A dirty tree
cannot produce a `FINALIZED` experiment; `--allow-dirty` produces a `DRAFT`.

## P0 — thesis-result integrity

### Fixed in the research runner

- A data revision is content-addressed from the original input SHA-256; a
  same-name replacement therefore produces a new revision ID.
- An experiment snapshots the exact Git commit and dirty-tree state, random
  seed, algorithm/prompt version, diagnosis thresholds, knowledge catalog
  hashes and environment package versions.
- Research mapping overrides and research ground truth are separate inputs.
  The export preserves AI prediction and final analysis mapping as separate
  columns; production human-review storage is not reused implicitly.
- Finalized artifact directories are immutable. Failed runs retain their
  submitted config and error record rather than being overwritten.
- Deterministic fixture reproduction is covered by tests and compares exported
  semantic and energy artifacts byte-for-byte.
- The agent regression and Local-LLM E2E artifacts no longer report a
  ``Hallucination Rate`` from route/tool checks. They now state factual
  exact-match coverage and accuracy only where a case declares a checkable
  expected fact. This prevents a structural pass from being presented as a
  semantic factuality result.

### Remaining P0 before publishing multi-project thesis claims

- A separate local `paper_research` protocol already contains a versioned
  semantic GT manifest and project-role metadata. It is not yet a portable
  BuildingAI research input: the new runner does not consume or enforce that
  split manifest, and the private provenance paths must not be copied into
  public experiment artifacts.
- The product `ProjectStore` still keeps only the current import metadata and
  current semantic result per project. Its `data_revision` counter invalidates
  state but is not an immutable historical dataset-revision store. Formal paper
  runs must use the research runner artifacts, not mutable project UI state.
- There is no repository-level project split registry that enforces
  development/validation/frozen-test/external-test separation. Do not report a
  final cross-project result until such a manifest is frozen and reviewed.
- There is no formal annotation protocol or independent reviewer workflow for
  semantic, diagnostic, or recommendation ground truth. A CSV GT input is
  versioned by the runner, but annotator agreement and dispute resolution still
  need a study-specific protocol.
- There is no blinded factuality/hallucination annotation set or qualified
  judge protocol for Agent answers. Do not make a paper claim about a
  hallucination rate until one is frozen separately from prompt development.
- Existing 66-case Agent regression, 52-case Local-LLM E2E and 12-query RAG
  artifacts are engineering/simulation fixtures, not real-BEMS ground-truth or
  human-evaluation evidence. They must not be relabelled as thesis benchmark
  scores. Their mutable `latest.json` outputs are also unsuitable as final
  paper artifacts until a versioned research runner is added for each suite.

## P1 — rigor and research efficiency

- Agent E2E artifacts record one documented Local-Qwen run. Repeated stochastic
  runs, mean/std/CI aggregation, and provider model digests are not yet a
  general experiment-matrix facility.
- Agent E2E task-success and regression task-success use different check
  compositions; compare them only within their own suite. E2E timing also does
  not yet separate the independently executed safety probe from total agent
  latency.
- RAG catalog hashes are captured, but frozen retrieval-query datasets and MRR/
  recall evaluation splits are not yet formalized.
- Drawing artifacts record configured detector metadata only; detector-weight
  SHA-256, confidence/image-size configuration, and a separate CV GT/split
  registry are still required for CV paper claims.
- Timestamp parsing and quality statistics are exported, but timezone/DST,
  resampling, interpolation and outlier-removal policies are not yet a unified
  research configuration registry. The current runner performs no hidden
  interpolation or imputation.
- Semantic backends can be selected by config, but a project-matrix baseline /
  ablation scheduler is not implemented. Do not alter source files per
  ablation; create explicit config variants and preserve all artifacts.
- `requirements.txt` is intentionally bounded but not lockfile-pinned. The
  experiment environment snapshot mitigates this; a lockfile is still advised
  before a long-lived performance or latency study.

## P2 — long-term improvements

- Add an append-only project-split and annotation registry with access control.
- Add bootstrap confidence intervals and repeated LLM evaluation orchestration.
- Add LaTeX table templates and domain-specific publication figure styles.
- Add a controlled anonymization manifest for blind-review exports.
- Capture GPU/CUDA driver details and provider model digests where available.

## Research artifact contract

Each experiment contains at least:

```text
config.json              immutable configuration and version snapshot
dataset_manifest.json    input identity, SHA-256, revision, range, quality
git.json                 code commit and dirty-tree state
environment.json         Python/platform/package snapshot
semantic_mapping.csv     AI prediction and final mapping kept separate
equipment_mapping.csv    stable equipment grouping
kpi_summary.csv          full-precision metric values and units
kpi_timeseries.csv       source-derived KPI samples and valid mask
energy_summary.json      raw chart inputs and engineering summaries
diagnostics.json         rule findings, evidence, thresholds and source points
results.json             machine-readable result summary
manifest.json            SHA-256 manifest for generated artifacts
reproduce.json           exact replay command and dataset hash
```

`scripts/generate_paper_figures.py <experiment-dir>` produces vector PDF and
300-dpi PNG figures from exported values, never from a GUI screenshot. It also
exports available CSV/Markdown tables. Missing capabilities are skipped rather
than fabricated.

## Required thesis workflow

```text
Data Freeze
  -> Ground-Truth Freeze
  -> Project Split Freeze
  -> Versioned Experiment Config
  -> Clean-commit Run
  -> FINALIZED Artifact
  -> Publication Figure/Table Export
  -> Paper Claim Mapping
```

Use `DRAFT` artifacts only for exploration. A paper table or figure must cite
its experiment ID, artifact manifest and dataset/GT version. Results shown in
the product UI are rounded for users and must never be copied as thesis values.

## Current boundary

The runner makes deterministic semantic/energy/diagnosis experiments
reproducible now. It does **not** by itself validate data leakage, establish
ground-truth quality, make LLM output deterministic, or convert human review
into CV/semantic research ground truth. Those remain explicit study-design
responsibilities.
