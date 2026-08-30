# BuildingAI

> AI-powered Building Energy Intelligence Platform.

BuildingAI is a local-first PyQt5 desktop application for turning heterogeneous BEMS/HVAC exports into reviewed, analytics-ready building data. It is designed for engineering analysis: it reads, analyzes, and recommends—it does not write to a building automation system.

## Highlights

- BEMS data import from CSV and multi-sheet Excel files
- HVAC semantic mapping using the research strict-8 taxonomy, V3 physical contradiction gate, and V4 structured evidence
- Physical validation, equipment organization, COP/KPI analysis, and diagnosis
- Fault and energy-saving opportunity workflow
- Pluggable LLM interface: Local Qwen (Ollama), OpenAI-compatible APIs, or custom endpoints
- Modern bilingual desktop UI: English / 中文, switchable without restart
- Offline-first operation: importing, mapping, analytics, and local project management work without an LLM
- Bounded Agent planning, evidence checking, reflection, scoped memory, source-aware RAG, and persisted traces
- Optional FastAPI and durable background-task interfaces; the desktop application remains local-first

## Architecture

```text
PyQt5 desktop UI → Services → Core / domain models → SQLite + Parquet
                                      ↘ optional LLM Provider interface
```

The deterministic product pipeline is:

```text
CSV/XLSX → quality metadata → semantic mapping → physical validation
         → ACCEPT / REVIEW / ABSTAIN → project storage → equipment / KPI / diagnosis
```

See [architecture notes](docs/architecture.md) for the semantic, persistence, and safety boundaries.

## Installation

BuildingAI supports Python 3.10+.

```powershell
git clone <your-fork-url>
cd building-ai-desktop
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```powershell
python app.py
```

The application starts normally with no LLM configured. Use **Settings** to select a provider and test its connection.

### Optional API and worker

```powershell
python -m pip install -r requirements-server.txt
python -m uvicorn building_ai.api.app:app --host 127.0.0.1 --port 8000
```

For Redis/RQ deployments, configure `BUILDING_AI_TASK_QUEUE=redis` and run the
worker entry point exposed by `building_ai.application.worker.run_task` through
your RQ worker process. The default local queue requires no Redis service.

## LLM configuration

The UI supports these providers:

| Provider | Required fields | Typical targets |
| --- | --- | --- |
| Local Qwen (Ollama) | Model, Ollama URL | Qwen via Ollama |
| OpenAI-compatible API | API Base URL, optional API Key, Model | OpenAI, DeepSeek, Qwen API, LM Studio, vLLM, Ollama-compatible servers |
| Custom / Other | API Base URL, optional API Key, Model | Any `/v1/chat/completions` compatible endpoint |

For a local Qwen model, install [Ollama](https://ollama.com/) separately and run:

```powershell
ollama pull qwen2.5:7b
```

API keys are saved only in the current user's operating-system configuration directory (not the repository) and are ignored by Git. You may also use environment variables; see [.env.example](.env.example).

## Screenshots

Add reviewed application screenshots to [`screenshots/`](screenshots/). The directory is intentionally empty in the source distribution so the README never presents generated or misleading UI images.

## Research background

`paper_research/` is the source of the current research baseline. The production application translates its selected strict-8 / V3 / V4 boundary into `building_ai` without importing the research directory at runtime. `building_ai/research/` retains an older C1–C8 compatibility mode only. See [research-to-product mapping](docs/research_to_product.md).

## Development

```powershell
pytest
```

The test suite covers imports, semantic results, project persistence, the C1–C8 adapters, COP/diagnosis logic, and the LLM configuration boundary.

## Evaluation

BuildingAI uses two complementary internal evaluation layers. They are not a public benchmark and their metrics must not be compared as though they were one suite.

1. **Deterministic Agent Regression Suite** — 66+ fast, repeatable cases for router/planner decisions, tool selection, read-only permissions, evidence and abstention logic, memory isolation, RAG retrieval, and prompt-injection boundaries. It deliberately does not require an LLM, so a near-zero LLM latency is expected. Run it with `python scripts/run_agentic_evaluation.py`; artifacts are written to `artifacts/evaluation/regression/`.
2. **End-to-End Local-Qwen Agent Evaluation** — 50+ natural-language cases using the configured Local Qwen/Ollama model through the complete bounded-agent path: router, planner, memory, tool registry, evidence checking, reflection, RAG where appropriate, and real final-answer generation. Run a representative smoke evaluation with `python scripts/run_e2e_agent_eval.py --quick`, or the full suite with `python scripts/run_e2e_agent_eval.py --full`. Artifacts and failure analysis are written to `artifacts/evaluation/e2e/`.

The E2E runner records real provider/model identity and request latency. It fails as an E2E run if Local Qwen was not actually called; token counts are reported as `N/A` when the local provider cannot expose them.

Example of a real multi-step evaluation trace:

```text
User: "Which machine is doing the worst?"
Route: equipment_analysis
Plan: get_equipment_kpis → get_diagnostic_findings
Evidence: PARTIAL
Reflection/re-plan: get_energy_timeseries
Evidence: SUFFICIENT
Local Qwen: evidence-constrained final explanation and next action
```

The E2E report retains the actual route, tools, reflection count, cited knowledge source IDs, safe answer summary, latency, and deterministic failure category for every case. It includes deliberate missing-data, unknown-equipment, prompt-injection, and tool-degradation cases; failed tool calls are therefore measured separately from safe recovery.

## Curated building-energy knowledge

BuildingAI includes a compact, source-attributed local knowledge catalog for
China, the United States, and Japan. It supports semantic/ontology concepts,
equipment and system relationships, engineering principles, O&M, energy-saving
measures, retrofit candidates, and government guidance. The catalog is not a
PDF-chat corpus: project tools determine what is happening in a building, while
knowledge explains concepts and suggests evidence or safe next checks.

The catalog retains original language and normalized multilingual concepts such
as `chilled_water_pump` / 冷冻水泵 / 冷水ポンプ. It contains only compact,
attributed summaries or metadata where redistribution is unclear—never copied
paid standards, vendor manuals, or user documents.

```powershell
python scripts/build_knowledge_base.py
python scripts/run_knowledge_rag_evaluation.py
```

See [knowledge sources and licensing](docs/knowledge_sources.md) for the source
registry, use restrictions, and rebuild behavior. The RAG evaluation measures
Top-1/Top-k retrieval for Chinese, English, and Japanese queries; it is an
internal regression measure, not a public benchmark.

## Current scope and limitations

- The application is a desktop Python project; packaging as an EXE is deliberately out of scope for this stage.
- KPI/COP and diagnosis are V1 engineering workflows and need site validation before operational changes.
- Long-running LLM research tasks do not yet have progress cancellation.
- BuildingAI has no BAS write path.
