# Historical v1.1 regression report

> Archived 2026-07-31 verification snapshot. Its test count is not the current
> main test count; consult the README and CI workflow for current-main checks.

Date: 2026-07-31

V1.1 final automated suite: **21 passed, 0 failed**. Module compilation passed. GUI smoke
test: **PASS** (eight pages constructed against schema v2; event loop exited cleanly).
Qt emitted an environment-specific missing Anaconda font-directory warning; it did not
prevent window construction or execution.

| Area | Fixture | Result | Evidence / limitation |
|---|---|---|---|
| Taxonomy | code-level expected 13 labels | PASS | `SemanticResult` rejects labels outside the preserved taxonomy |
| Direct Prompt variants | code-level definitions | PASS | V1–V4 retained; formal research batch still hard-codes Direct V1 for Full Model |
| Data matching | synthetic full-width/ASCII column names | PASS | preserved matching helper produces the same normalization |
| ACCEPT/REVIEW/ABSTAIN | synthetic model objects | PASS | adapter follows abstention/gate/suspicious state; no new confidence threshold |
| Human verification | temporary SQLite | PASS | AI label and human final label round-trip separately |
| Import metadata | synthetic CSV/XLSX | PASS | time range, interval, missing data and sheet selection covered |
| Offline semantic pipeline | three synthetic HVAC columns | PASS | explicit supply/return tokens map; ambiguous point ABSTAINS |
| Agent safety | synthetic ABSTAIN point | PASS | time-series tool refuses unverified ambiguous semantics |
| Point identity | temporary SQLite, same name across files/sheets | PASS | three records and point-specific review remain distinct |
| Schema migration | synthetic V1 SQLite database | PASS | AI prediction and review preserved under schema v2 |
| Research client compatibility | mocked product LLM client | PASS | OpenAI-shaped `choices[0].message.content` is provided |
| Research unit boundary | frozen/mocked C6 response | PASS | unit is read from research C6, not product `infer_unit` |
| Generic quantity safety | synthetic Japanese/Chinese/English headers | PASS | quantity retained but no default heat-source ACCEPT |
| C1–C8 research result parity | no controlled LLM transcript fixture | NOT TESTED | Real-world regression pending |
| Unit-combination selection | no frozen legacy fixture | NOT TESTED | Requires captured input, LLM response and expected `unit_db` |
| Legacy heat-source COP | no small validated old-result fixture | NOT TESTED | New interface keeps formula/range, but full branch equivalence is unverified |
| Terminal COP | not migrated | NOT TESTED | Pending behavior-preserving extraction |
| Load/weather correlation | no small validated old-result fixture | NOT TESTED | Interface only/minimal implementation |

## Frozen regression fixture v1

`tests/fixtures/regression_v1/` contains a small synthetic multilingual CSV, one frozen
C1–C8 response, expected product-offline semantics, and expected C6 units. It covers
English, Japanese, and Chinese names, explicit heat-source/terminal evidence, and an
unknown point. It is intentionally marked `claims_real_world_equivalence: false`.

This is the first deterministic regression harness, not proof of equivalence to a real
building or to every legacy algorithm branch.

## Real-world regression pending

The old project contains real Excel/CSV datasets, but no small, self-contained fixture with
frozen LLM responses and an authoritative expected semantic/slot/unit/COP/load result was
identified. Running a live LLM would not be a deterministic regression test.

The next regression corpus should contain:

1. A de-identified small DataFrame with representative Japanese/Chinese/English point names.
2. Recorded provider/model, Direct Prompt version, temperature 0 and seed 0.
3. Frozen LLM JSON responses for slot, semantic and unit calls.
4. Expected `ai_roles`, gate status, suspicious flag, C1–C8 details, unit database, COP and
   load outputs from a named old commit.
5. Explicit classification of every difference as expected or unexpected.
