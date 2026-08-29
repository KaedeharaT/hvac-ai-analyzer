# BuildingAI Agentic Platform Roadmap

## Architecture

`PyQt / FastAPI -> ApplicationContext -> deterministic BEMS analysis core` with a
bounded, read-only Agent Runtime and SQLite-backed task persistence.  All
presentations must consume the same project, semantic, equipment, energy,
diagnosis and opportunity services.

## Current verified baseline

| Area | Status | Production evidence | Test/runtime evidence |
|---|---|---|---|
| Desktop and analysis core | PASS | `building_ai/ui`, `building_ai/services` | existing pytest suite |
| FastAPI | PASS | `building_ai/api/app.py` | uvicorn health/project/task smoke |
| Persistent local task | PASS | `building_ai/application/tasks.py` | Project 7 task succeeded |
| Router / bounded plan model | PARTIAL | `building_ai/agent_runtime.py` | model construction only |
| Background local dispatch | PARTIAL | `TaskService.submit_background` | requires isolated task smoke |

## Backlog and acceptance criteria

All entries below remain **FAIL** until the acceptance runner has a real test
or runtime artifact for it.  No interface/class alone is evidence.

### Task infrastructure
- [ ] Redis Queue Adapter — production adapter plus local fallback test.
- [ ] WAITING_REVIEW / Review Resume — persisted review payload and resume test.
- [ ] Retry — bounded retry count / error persistence test.
- [ ] Timeout — task/tool timeout test.
- [ ] Concurrent idempotency — concurrent submit returns one task.

### Agent runtime
- [ ] Multi-step Planner — dynamic project-aware tool sequence runtime trace.
- [ ] Tool Registry — Pydantic schemas, permission and timeout for every tool.
- [ ] Evidence Checker — SUFFICIENT/PARTIAL/INSUFFICIENT.
- [ ] Reflection / Re-plan — persisted trace `PARTIAL -> REPLAN -> SUFFICIENT`.

### Memory
- [ ] Conversation Memory — pronoun follow-up real runtime case.
- [ ] Project Memory — scoped facts/findings/analysis summary.
- [ ] Cross-project isolation — no memory or equipment leakage.

### RAG and knowledge
- [ ] Ingestion, chunking, metadata and retrieval.
- [ ] `search_knowledge` Agent tool and citation.
- [ ] Knowledge source registry and `docs/knowledge_sources.md`.
- [ ] Source tracks: Haystack, Brick, Open223, DOE FEMP, NREL BCL, EnergyPlus.

### Observability and evaluation
- [ ] Persistent Agent Trace, Tool Trace, LLM Trace, Evidence Trace.
- [ ] `GET /traces`, `GET /traces/{trace_id}`.
- [ ] Eval dataset (>=60 meaningful cases) and deterministic runner.
- [ ] Routing, tool selection, grounding, hallucination, abstention and failure metrics.

### Security and infrastructure
- [ ] READ_ONLY / WRITE / DANGEROUS tool permission enforcement.
- [ ] User/BEMS/RAG prompt-injection tests.
- [ ] Trace/log secret redaction tests.
- [ ] Docker / Redis compose plus environment-specific runtime status.

## Last validation

2026-08-30: `python -m pytest` completed with **76 passed**.  Uvicorn health
and Project 7 task endpoint were previously exercised.  The acceptance script
is authoritative for subsequent status.
