# Roadmap

> Historical planning document. Status labels below describe the migration plan
> at the time it was written, not the current product capability set. See the
> README and architecture documentation for current scope.

## V0 — Research code migration

- Preserve C1–C8 source, taxonomy, Direct Prompt variants, deterministic settings, and tests.
- Add regression fixtures before making internal algorithm changes.

## V1 — Mapped-lite (current)

- Persistent projects, local data import, quality metadata, semantic mapping, units,
  equipment hints, physical validation, ACCEPT/REVIEW/ABSTAIN, and human verification.
- Complete UI background execution, point provenance, and real-world regression corpus.

## V2 — Clockworks-lite

- Standardized equipment relationships, scalable time-series queries, FDD, diagnostics,
  root-cause evidence, analytics evaluation, and stronger weather normalization.

## V3 — ARIA-lite

- Building AI Agent over trusted tools, retrieval over project documents, grounded
  explanations, conversation audit, and role-based recommendations.

## V4 — Optimization shadow mode

- System identification, forecast models, constrained optimization/MPC and RL comparison,
  safety envelopes, counterfactual replay, and shadow-mode validation.

## V5 — Building automation integration

- Read-only protocol gateways first. Any later write integration requires fail-safe design,
  physical validation, approval workflows, audit, rollback, and site-specific commissioning.
