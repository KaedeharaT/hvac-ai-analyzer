# Competitive redesign QA

This record validates the product redesign without publishing private building data. Exact private project names, source filenames, point names, timestamps and values are intentionally omitted.

## Product journeys

### Full-data local project

- Project open and persisted analysis reload: PASS
- Semantic mappings and four equipment contexts available: PASS
- Energy, power, temperature, ΔT, COP, daily profile, heatmap, weather relation, equipment comparison and period comparison: AVAILABLE
- Deterministic findings and evidence-backed opportunities: AVAILABLE
- Dashboard → finding → diagnostic detail: PASS
- Equipment → energy / AI / drawing navigation: PASS
- Single-Agent project investigation: PASS
- Multi-Agent project investigation with Coordinator, Data Analyst, HVAC Expert and Reviewer: PASS

No source data or identifying project metadata was copied into this document or the screenshot artifacts.

### Partial-data local project

- Project open: PASS
- Temperature trend: AVAILABLE
- Equipment KPI, COP, power and energy: UNAVAILABLE because required confirmed signals are absent
- False zero KPI or empty fabricated chart: NOT OBSERVED
- Capability reason presentation: PASS

### Privacy-safe drawing journey

A generated geometric layout and a fake detector adapter were used only to exercise the production persistence and UI workflow.

- Import → detection record → confirmation → equipment mapping: PASS
- Confirmed equipment overlay: PASS
- Operational attention comes from deterministic BEMS findings, not CV: PASS
- Drawing → equipment detail: PASS

## Visual QA

The repository-local capture helper exercised Overview, Equipment, Energy, Diagnostics, AI Assistant, Knowledge Base and Drawings at 1280×720, 1440×900 and 1920×1080. A second run used 125% Qt scaling. Captures are stored under the gitignored `artifacts/visual_qa/` directory and are not public product evidence.

## Evidence and action boundaries

- “Passed check” means an applicable deterministic rule ran and did not trigger; it is not a complete health claim.
- Possible causes and recommended checks remain separate from measured project findings.
- Energy impact is displayed as unavailable when no supported calculation exists.
- Period comparison is measured comparison, not claimed savings.
- Agent tools remain read-only in both Single-Agent and Multi-Agent modes.
- Knowledge evidence cannot create a project finding.
- Only human-confirmed drawing associations receive equipment context.

## Current product limitations

- The application remains a local/project-level research product, not a live portfolio or BAS platform.
- There is no work-order execution, tariff/cost model, verified savings workflow or control path.
- Drawing status overlays require a human-confirmed equipment association.
- Natural-language quality remains provider/model dependent; technical trace is required when assessing Agent behavior.
- 150% DPI was not treated as a release gate in this environment; 100% and 125% were directly captured.
