# Optional Multi-Agent Runtime

BuildingAI retains the established Single-Agent Runtime as its default and
baseline. The optional `AGENT_MODE=multi` runtime is a coordinator-led,
evidence-bounded alternative for research comparison and complex questions. It
does not replace deterministic BEMS analytics, diagnosis rules, storage, RAG
retrieval, or drawing detection.

## Roles and evidence boundary

```text
User → Coordinator → role-bounded specialists → Reviewer → Coordinator answer
```

- **Coordinator** decomposes the request and collects structured results; it
  does not own the full business-tool surface.
- **Data Analyst** reads project equipment, KPI, time-series, capability, and
  deterministic diagnostic evidence and returns a `DataEvidencePacket`.
- **HVAC Expert** interprets only the supplied project packet. When evidence
  is incomplete it requests specific evidence instead of inventing a finding.
- **Knowledge Agent** retrieves general reference material only. Its
  `KnowledgeEvidencePacket` is never a statement of a current-project fact.
- **Drawing Agent** reads only human-confirmed drawing associations; predicted
  and rejected detections are excluded from engineering evidence.
- **Reviewer** checks evidence availability, conflicts, unsupported project
  claims, and drawing confirmation. It can approve, request more evidence,
  flag a conflict, or require abstention.

Every inter-agent message is a Pydantic schema (`AgentTask`, `AgentResult`,
and typed evidence or review packets), rather than unbounded natural-language
handoff. Evidence is treated as data, never as instructions.

## Permissions and safety

Each specialist has a separate tool allowlist. Data, knowledge, and drawing
tools are all read-only. The HVAC Expert and Reviewer have no direct business
tool permissions. In particular, no multi-agent role can confirm drawing
detections, change semantic mappings, modify diagnoses, edit project data, or
control a BAS.

The Coordinator uses ordered delegation by default. It may only parallelize
tasks with no evidence dependency; the initial runtime intentionally keeps
dependent engineering interpretation after project-data collection.

## Product and research configuration

`AGENT_MODE=single` remains the default, preserving current UI and API
behavior. `AGENT_MODE=multi` can be selected in Settings or passed to
`POST /agent/chat` as `agent_mode: "multi"`; omitted API clients remain on the
Single-Agent mode.

Multi-agent traces retain parent/child links, role, task, tools, status,
latency, and versioned deterministic role policies. Research experiment snapshots record `agent_mode`, architecture
version (`multi-v1`), role list, coordination and review policies, maximum
review rounds, and read-only tool permissions. The frozen Single-Agent and
Multi-Agent deterministic regressions therefore use the same 66 cases while
recording coordination overhead separately.

## Interpretation

Multi-Agent Runtime is an experimental, optional architecture. It is not
assumed to outperform the Single-Agent baseline: comparisons should report
task success, evidence grounding, abstention, tool and LLM calls, latency, and
failure recovery on the same frozen dataset.
