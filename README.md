# BuildingAI Desktop

> AI-assisted building energy and HVAC operational data analysis platform.

BuildingAI is a local-first PyQt5 desktop application for turning heterogeneous BEMS/HVAC exports into reviewed, analytics-ready building data. It is designed for engineering analysis: it reads, analyzes, and recommends—it does not write to a building automation system.

## Highlights

- BEMS data import from CSV and multi-sheet Excel files
- HVAC semantic mapping using the research strict-8 taxonomy, V3 physical contradiction gate, and V4 structured evidence
- Physical validation, equipment organization, COP/KPI analysis, and diagnosis
- Fault and energy-saving opportunity workflow
- Pluggable LLM interface: Local Qwen (Ollama), OpenAI-compatible APIs, or custom endpoints
- Modern bilingual desktop UI: English / 中文, switchable without restart
- Offline-first operation: importing, mapping, analytics, and local project management work without an LLM

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

## Current scope and limitations

- The application is a desktop Python project; packaging as an EXE is deliberately out of scope for this stage.
- KPI/COP and diagnosis are V1 engineering workflows and need site validation before operational changes.
- Long-running LLM research tasks do not yet have progress cancellation.
- BuildingAI has no BAS write path.
