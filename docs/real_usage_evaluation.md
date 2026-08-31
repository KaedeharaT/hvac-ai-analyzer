# Real usage evaluation

This note records a local, user-journey evaluation performed against BuildingAI
V1.2.0. It is not a benchmark and it does not include raw BEMS exports, private
drawing files, model weights, or project databases.

## Journey A — complete local BEMS project

1. Opened a persisted local BEMS project in the desktop application.
2. Restored its imported time series, semantic mappings, equipment groups,
   deterministic analysis, diagnostics, and opportunities.
3. Reviewed Energy Analysis with the available energy, power, temperature,
   delta-T, COP, daily-profile, heatmap, weather, ranking, and period charts.
4. Queried the AI Assistant for project status, equipment performance, an
   equipment-specific delta-T issue, a follow-up, and the equipment list.

The complete journey produced four heat-source equipment groups, validated
equipment KPIs, deterministic low-delta-T findings, and site-validation-bound
recommendations. The Assistant used read-only project tools; it retained an
equipment focus for a follow-up and used knowledge retrieval for the
delta-T-improvement question. Project-specific numerical findings stay local.

## Journey B — partial historical BEMS workbook

A separate local historical workbook was imported, semantically mapped, and
analysed. It provided a timestamp and mapped temperature series but no complete
heat-source group with supply temperature, return temperature, flow, and power.

- Temperature trend remained available.
- Energy, power, COP, delta-T, ranking, and period comparison remained
  unavailable rather than being inferred.
- The Energy Analysis page presents the unavailable analysis with its data
  requirement when it is selected.
- The Assistant states that COP cannot be reliably calculated and identifies
  the remaining available analysis.

## Drawing Intelligence journey

Using the repository's public synthetic demonstration image, a local review
journey imported the drawing, saved two test detections, confirmed one,
rejected one, and linked the confirmed object to an existing equipment ID.
After a restart, the drawing, review states, and association persisted. The
read-only drawing-location tool returned the confirmed location and abstained
when an equipment had no confirmed association.

This workflow validates the product boundary, not detector accuracy. Real YOLO
model weights and real building drawings remain local and are not distributed.

## Knowledge-base journey

Chinese, English, and Japanese queries returned multilingual catalog entries
with country/category metadata and a source link. Two sampled source links
opened successfully; one older EnergyPlus PDF link returned HTTP 404 and is a
known catalog-link maintenance issue.

## Known limitations found by this evaluation

- Project-data answers are intentionally deterministic and evidence-bounded;
  the local LLM is principally used for general engineering conversation, not
  to override project evidence.
- The Assistant's equipment-list answer currently includes KPI/finding context
  in addition to the requested list, so it is more verbose than necessary.
- Some historic workbooks remain review-heavy and therefore cannot form
  complete equipment KPI groups without human semantic confirmation.
- Knowledge-source URLs require periodic link checking.
- Drawing-object association is manual. A detection is not an engineering fact
  until a reviewer confirms it.
