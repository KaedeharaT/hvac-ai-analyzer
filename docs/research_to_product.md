# Research-to-product mapping

`paper_research` is the evidence and experiment source. `building_ai` is a
standalone product implementation and has no runtime import from that directory.

## Selected production boundary

The paper's evaluated formal baseline is the strict eight-label Direct prior
with the conservative V3 explicit non-target quantity gate. V4 provides the
latest structured equipment/loop/quantity parser. Its development selection
explicitly excludes group reassignment and temporal intervention; the V4
equipment-aware configuration has not run the Formal Test. BuildingAI therefore
uses V3's gate for semantic correction, and V4 evidence for traceability,
equipment grouping, validation, and KPI readiness—not unvalidated temporal or
group overrides.

| Research source | Product implementation | Product use |
| --- | --- | --- |
| `src/bems_v2/ingest.py` | `services/import_service.py`, `core/preprocessing.py` | CSV/XLSX, timestamps, Unicode headers |
| `src/bems_v2/taxonomy.py`, `semantic_protocol.py` | `models/semantic_result.py` | Strict 8 active labels; legacy 13 read compatibility |
| `src/bems_v3/core.py` | `core/research_semantics.py` | Explicit non-target contradiction gate |
| `src/bems_v4/core.py` | `core/research_semantics.py` | Structured equipment, loop, medium, quantity and role evidence |
| `src/bems_v2/llm.py` | `llm/BaseLLMProvider`, `LLMManager` | Provider-neutral Direct JSON prior |
| V4 equipment identifiers/groups | `services/equipment_service.py` | Evidence-backed heat-source bindings for COP/KPI |
| `power_energy_features.py` | `core/cop_engine.py`, `services/analytics_service.py` | Deterministic power/COP readiness and KPI calculation |

## Deliberately not productized

- Ground truth, LOPO/Formal-Test evaluation, macro-F1, ablations, frozen caches,
  experiment manifests, and paper-output scripts are research-only.
- V4 group consistency reassignment and temporal relation correction are not
  inference interventions: the selected configuration reports no independent
  gain, and the equipment-aware variant lacks a Formal-Test run.
- The old C1–C8 helper is retained as `legacy_c1c8` only for compatibility with
  earlier stored research comparisons. The default product path does not call it.
