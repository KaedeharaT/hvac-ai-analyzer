# BuildingAI Agentic Platform Autopilot

You are continuing development in the current workspace.  The only authoritative
state is `AGENTIC_PLATFORM_ROADMAP.md`, `scripts/run_agentic_acceptance.py`,
`artifacts/acceptance/latest.json`, and Git history.

At startup read those files, inspect Git status and recent commits, then select
one complete dependency-ordered batch addressing current acceptance FAIL items.
Implement production code and meaningful tests; use targeted tests during work.
At the end of a completed batch run full pytest and acceptance.  Do not lower
acceptance standards, fake evidence/metrics, delete hard cases, hard-code test
projects/devices, or regress passing capabilities.  Prefer the existing shared
application core.  Commit stable changed code, but a commit is only a checkpoint.

Continue to the next batch when time allows.  Before ending update the roadmap
and leave `artifacts/acceptance/latest.json` current.

Output only:
STATUS
CURRENT_COMMIT
PASS
FAIL
ENVIRONMENT_BLOCKED
PROGRESS_MADE
NEXT_BATCH
