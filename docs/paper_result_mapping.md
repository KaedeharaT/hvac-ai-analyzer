# Paper result mapping

Use this append-only template when a thesis table, figure, or numerical claim
is finalized. Do not copy values from the UI.

| Paper location | Claim / caption | Experiment ID | Dataset revision | GT / mapping version | Git commit | Artifact file | Manifest SHA-256 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Table / Figure | _fill after FINALIZED run_ | `EXP-...` | `r-...` | `gt-v...` / `sem-v...` | `<commit>` | `results.json` | `<hash>` |

For each claimed metric also retain its denominator (`n`), aggregation method,
uncertainty treatment, and any exclusion rule in the experiment config.
