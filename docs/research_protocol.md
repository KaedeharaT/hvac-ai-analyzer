# Research Protocol Phase 2

Use `research_protocol: "formal"` only after data, ground truth, and the
project split are frozen. Formal configurations must provide:

- `split_registry_path`: a frozen `project_split` manifest with one assignment
  (`development`, `validation`, `frozen_test`, or `external_test`) per
  anonymised project and its input SHA-256.
- `ground_truth_path` and `ground_truth_freeze_path`: a frozen GT manifest
  whose SHA-256 matches the annotation CSV.

Optional `evaluation_dataset_freeze_path` records a frozen Agent/RAG evaluation
dataset. CV experiments additionally require a local model path plus a frozen
`cv_split` manifest; artifacts record the model SHA-256, classes, confidence
threshold, and image size without publishing the weight.

`python scripts/run_research_matrix.py --matrix matrix.json` executes explicit
baseline and ablation configurations and aggregates numeric metrics over
declared seeds as mean, standard deviation, min, max, and n. A matrix never
edits source code to disable a component. Matrix and experiment outputs remain
ignored research artifacts.

Agent traces can be exported with `export_agent_traces(...)`; the export keeps
route, plans, tools, evidence state, source identifiers, LLM metadata, latency,
and multi-agent hierarchy, but intentionally excludes user-answer prose.
