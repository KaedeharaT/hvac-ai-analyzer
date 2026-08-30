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
- [x] Redis Queue Adapter — production adapter plus local fallback test.
- [ ] WAITING_REVIEW / Review Resume — persisted review payload and resume test.
- [x] Retry — bounded retry count / error persistence test.
- [x] Timeout — task/tool timeout test.
- [ ] Concurrent idempotency — concurrent submit returns one task.

### Agent runtime
- [ ] Multi-step Planner — dynamic project-aware tool sequence runtime trace.
- [ ] Tool Registry — Pydantic schemas, permission and timeout for every tool.
- [ ] Evidence Checker — SUFFICIENT/PARTIAL/INSUFFICIENT.
- [ ] Reflection / Re-plan — persisted trace `PARTIAL -> REPLAN -> SUFFICIENT`.

### Memory
- [x] Conversation Memory — pronoun follow-up real runtime case.
- [x] Project Memory — scoped facts/findings/analysis summary.
- [x] Cross-project isolation — no memory or equipment leakage.

### RAG and knowledge
- [x] Ingestion, chunking, metadata and retrieval.
- [x] `search_knowledge` Agent tool and citation.
- [ ] Knowledge source registry and `docs/knowledge_sources.md`.
- [ ] Source tracks: Haystack, Brick, Open223, DOE FEMP, NREL BCL, EnergyPlus.

### Observability and evaluation
- [x] Persistent Agent Trace, Tool Trace, LLM Trace, Evidence Trace.
- [ ] `GET /traces`, `GET /traces/{trace_id}`.
- [x] Eval dataset (>=60 meaningful cases) and deterministic runner.
- [x] Routing, tool selection, grounding, hallucination, abstention and failure metrics.

### Security and infrastructure
- [x] READ_ONLY / WRITE / DANGEROUS tool permission enforcement.
- [x] User/BEMS/RAG prompt-injection tests.
- [x] Trace/log secret redaction tests.
- [ ] Docker / Redis compose plus environment-specific runtime status.

## Last validation

2026-08-30: The bounded runtime now rejects user instruction overrides before
tool or LLM dispatch, treats BEMS/RAG content as untrusted data, enforces its
READ_ONLY tool boundary, and redacts secrets recursively before trace storage.
Actual LLM invocation metadata is persisted without prompt or response content.
The deterministic runtime evaluation includes 68 generic cases covering route,
tool selection, grounding, hallucination avoidance, abstention, and failure
safety. Full validation completed with **90 passed**; acceptance reported
**26 PASS / 0 FAIL / 0 ENVIRONMENT_BLOCKED**.

2026-08-30: Redis/RQ dispatch now uses an importable worker entry point while
local development remains on the in-process queue. Retry count and error
persistence are covered by the task state-machine suite. Conversation and
project memory are scope-isolated; the knowledge service chunks sources and
returns provenance-bearing citations through the Agent runtime. Full
validation completed with **84 passed**; acceptance reported **21 PASS / 5
FAIL / 0 ENVIRONMENT_BLOCKED**. The acceptance script remains authoritative
for subsequent recorded counts.
