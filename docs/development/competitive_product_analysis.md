# Competitive product analysis

This document records the public-product research used to redesign BuildingAI. It is an internal design reference, not a claim that BuildingAI has commercial-platform scope. Research was performed against official product pages and public product updates on 2026-09-03.

## Comparison matrix

| Dimension | BuildingAI boundary | OpenBlue / OBI | Clockworks Analytics | Facilio | DK-CONNECT |
|---|---|---|---|---|---|
| Product positioning | Local-first BEMS and drawing analysis with a deterministic engineering core and read-only AI | Unified building-data and AI intelligence layer spanning energy, equipment and faults | Equipment-centric FDD and condition-based maintenance | Portfolio operations / CMMS with role-specific AI assistants | Remote multi-site equipment monitoring, control and energy management |
| Main navigation | Project analysis workflow | Portfolio/building/equipment and cross-domain conversational access | Diagnostics, equipment, data and task workflows | Operational modules with AI available across them | Property, equipment list, layout, energy and reports |
| Dashboard | Project-level evidence and readiness | Unified operational/energy/equipment overview | Summary widgets drive cross-filtered diagnostic tables | Portfolio visibility and exception-oriented operations | Property and equipment state overview |
| Data onboarding | CSV/Excel import, semantic review and equipment discovery | Automated normalization and semantic modeling | Deployment timeline, connectivity and QA summaries | Integration-led onboarding | Gateway/controller setup |
| Equipment model | Evidence-backed heat-source groups | Unified asset/equipment context | Equipment is the primary unit for diagnostics and related faults | Asset lifecycle context | Equipment list and individual operation panel |
| Energy analytics | Capability-driven energy, power, temperature, ΔT and COP | Portfolio energy, comfort and optimization views | Diagnostic data charts and saved/shared chart templates | Energy monitoring and portfolio analytics | Actual/target energy, trends and exported interval data |
| Diagnostics | Deterministic rule findings | FDD, prioritization and root-cause assistance | Individual fault report with occurrence period, impact and related equipment | Operational exception workflows | Equipment alarm/state monitoring |
| Fault prioritization | Evidence strength and rule severity only | Prioritized faults and alarms | Energy, comfort and maintenance priorities | Role/workflow priority | State/alarm filtering |
| Root cause | Possible causes kept separate from findings | Correlates energy, faults and equipment | Related equipment faults and diagnostic context | Cross-module operational context | Operator interpretation of equipment state/trends |
| Recommendations | Deterministic finding → bounded checks; no fabricated savings | Recommendations and approved automation | AI summary and recommended resolution actions | Proposed/automated actions within operations workflows | Control and energy-management actions |
| AI assistant | Read-only Single/Multi-Agent over project tools and knowledge | Context-aware natural-language interface with guided prompts | Natural-language data exploration and orchestrated specialist agents | Contextual FM Copilot and purpose-built agents | Not the central interaction model |
| Evidence | Project Evidence and Reference Material are separated | Unified building data used by AI applications | Diagnostic calculations, notes, costs and charts | CMMS/asset/workflow record | Measured equipment and energy data |
| Knowledge | Curated local multilingual source registry | Vendor/domain intelligence layer | Global knowledge used to contextualize diagnostics | Documentation and operational records | Product/service guidance |
| Drawing / layout | Confirmed CV detections can be linked to equipment | Building/equipment navigation | Equipment relationships | Asset/location context | Floor-plan layout with equipment icons and state |
| Report | Markdown/HTML analysis report from current deterministic results | Portfolio/performance reporting | Diagnostic reports and task summaries | Dashboards and operational reporting | Energy/report export with preview |
| Workflow | Investigate and recommend only | Can execute approved workflows | Diagnostic-to-task workflow | CMMS actions, dispatch and approvals | Remote monitoring and control |
| Human approval | Semantic and drawing review; Agent remains read-only | Human-in-the-loop for approved workflows | Operator/task workflow | Approval packets and operational authorization | User permissions and operator control |
| Missing-data UX | Capability reasons and semantic review status | Onboarding/data-layer visibility | Connectivity and QA categories such as not reporting/flatlined/out of range | Exception handling across workflows | Connection/equipment state feedback |
| Drill-down | Project → equipment → finding/evidence/drawing | Portfolio → building → equipment/fault | Equipment → diagnostic → charts/related faults | Portfolio → asset/workflow | Property → layout/equipment panel |
| Export | Research artifacts plus bounded product report | Enterprise reporting/APIs | Shared charts and reports | Reports/dashboards | Interval data and report export |
| Research support | Versioned experiments, baselines, ablations and replay | Not a public research-platform focus | Physics/AI product validation, not a public experiment runner | Not a public research-platform focus | Not a public research-platform focus |

## ADOPT

- Equipment-centric navigation and drill-down, because equipment is the stable bridge between semantics, KPIs, findings and drawings.
- Readiness-first onboarding, so missing analysis is explained by mapping and signal coverage rather than an empty chart.
- Diagnostics as a filterable workbench with a structured detail view, supporting evidence, possible causes and recommended checks.
- Passed checks only when a deterministic rule was actually evaluated and did not trigger; this is not a health assertion.
- Context-aware AI entry points and guided questions tied to the current project/equipment/finding.
- Progressive disclosure: operational summary first, deeper engineering charts and trace details second.
- Cross-navigation between dashboard, equipment, diagnostics, energy, drawings, AI and knowledge.
- Report export based on deterministic results and provenance, not prose-only screenshots.

## ADAPT

- DK-CONNECT-style layout linkage becomes a read-only overlay on human-confirmed drawing mappings. BBox evidence remains CV evidence; operational state comes from BEMS/diagnostics.
- Clockworks-style impacts are shown only when BuildingAI has a supported measured calculation. Otherwise the UI explicitly says that energy impact is unavailable.
- Commercial priority scoring is replaced by the existing rule severity/evidence boundary; no tariff or maintenance-cost score is invented.
- AI workflow is presented as an investigation result (evidence checked, possible causes, next checks, verification) while the underlying Single/Multi-Agent trace remains available only in technical details.
- Period comparison is described as measured comparison, never as verified savings without baseline/intervention metadata.
- Facilio/OpenBlue action patterns are limited to analysis, explanation and recommendation. BuildingAI does not execute work orders or building control.

## REJECT

- BACnet/Modbus/OPC gateways and live controls: outside the current CSV/Excel and image input boundary.
- CMMS/ERP/work-order write actions: BuildingAI Agent tools remain read-only.
- Automatic fault monetization: no reliable tariff, labor-cost or avoided-failure model exists in the current platform.
- Autonomous setpoint or schedule changes: incompatible with the deterministic, human-reviewed research boundary.
- Automatic BEMS-to-drawing matching and topology reconstruction: current associations are human-confirmed.
- Portfolio-scale claims and commercial SLA language: not supported by the present research prototype.

## Design decisions for BuildingAI

The redesign keeps project, equipment and evidence as the persistent user context. Summary cards answer what needs attention; tables support scanning and selection; detail panels explain evidence and limitations. AI is available from the active context, but never replaces the deterministic analysis. Research services and experiment artifacts remain UI-independent.

## Official references

- Johnson Controls, OpenBlue Intelligence: https://openblue.johnsoncontrols.com/openblue-intelligence
- Clockworks Analytics, Redesigned Diagnostic Report: https://clockworksanalytics.com/redesigned-diagnostic-report/
- Clockworks Analytics, Onboarding design updates: https://clockworksanalytics.com/onboarding-design-updates/
- Facilio, Atom AI Suite: https://facilio.com/ai-suite/
- Daikin, DK-CONNECT layout view: https://www.ac.daikin.co.jp/solution/dkconnect/service/modal/03
- Daikin, DK-CONNECT energy visualization: https://www.ac.daikin.co.jp/solution/dkconnect/service/modal/18
