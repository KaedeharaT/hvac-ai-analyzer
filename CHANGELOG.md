# Changelog

> Release entries are immutable records for their tagged versions. The v1.2.0
> validation below applies to that release tag; `main` may contain later
> documentation and UI improvements. See the README and CI status for current-main verification.

## v1.2.0

### Added

- Drawing Intelligence for project-local drawing import and review.
- A YOLOv8 drawing-detector adapter behind a replaceable detector abstraction.
- Persistent drawing objects, human review, and equipment-to-drawing association.
- Read-only Agent tools for confirmed drawing evidence.
- Public Drawing Intelligence documentation.

### Existing platform capabilities

- Heterogeneous BEMS semantic understanding and equipment discovery.
- Energy and KPI analytics, HVAC diagnostics, and actionable recommendations.
- Agent Tool Calling, evidence checking, reflection, project and conversation memory.
- Multilingual RAG, traceability, evaluation, FastAPI, and background-task support.

### Evaluation

- pytest: 110 passed.
- Agentic Acceptance: 26 PASS / 0 FAIL.
- GitHub CI: PASS.

### Safety and boundaries

- Agent tools remain read-only.
- YOLO detections require human confirmation before equipment association.
- Model weights are not distributed.
- Private building drawings are not included.
