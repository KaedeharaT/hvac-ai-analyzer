"""Run a small non-sensitive Local-LLM smoke through MultiAgentRuntime.

This is deliberately not a benchmark.  It verifies that the optional runtime
can use the configured local provider for a general knowledge turn while also
coordinating a project-evidence turn against an in-memory fixture.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from building_ai.config import Settings
from building_ai.e2e_evaluation import E2ELLMController, E2EProjectTools
from building_ai.knowledge import KnowledgeService
from building_ai.llm import LLMManager
from building_ai.multi_agent_runtime import MultiAgentRuntime
from building_ai.observability import TraceStore
from building_ai.storage import Database


def main() -> int:
    settings = Settings.load()
    provider = LLMManager(settings).get_provider()
    connected, reason = provider.test_connection()
    if getattr(provider, "provider_id", None) != "local_llm" or not connected:
        print(json.dumps({"status": "ENVIRONMENT_BLOCKED", "reason": reason}, ensure_ascii=False))
        return 2
    with TemporaryDirectory(prefix="buildingai-multi-local-smoke-") as directory:
        database = Database(Path(directory) / "smoke.sqlite3")
        tools = E2EProjectTools()
        controller = E2ELLMController(provider, settings.model)
        context = SimpleNamespace(
            database=database,
            agent=SimpleNamespace(tools=tools),
            agent_controller=controller,
            ensure_project_loaded=lambda _: None,
        )
        KnowledgeService(database).ingest(
            "HVAC Operations Handbook", "evaluation://multi-agent-smoke",
            "Low chilled-water delta T checks include bypass flow, pump differential-pressure reset, terminal flow, and coil heat transfer.",
            "Operations guidance", {"fixture": "non-sensitive"}, chunk_size=80,
        )
        runtime = MultiAgentRuntime(context)
        general = runtime.run("What should I inspect when chilled-water delta T is low?", "smoke-project", "multi-smoke")
        project = runtime.run("Which equipment has the lowest COP?", "smoke-project", "multi-smoke")
        def summary(result):
            trace = TraceStore(database).get(result.trace_id)
            return {
                "trace_id": result.trace_id,
                "answer": result.answer,
                "grounded": result.grounded,
                "abstained": result.abstained,
                "roles": [item["agent_role"] for item in trace.get("multi_agent", {}).get("results", [])],
                "tool_calls": [item["tool"] for item in trace.get("tool_calls", [])],
                "real_llm_calls": len(trace.get("llm_calls", [])),
            }
        print(json.dumps({"status": "PASS", "provider": provider.provider_id, "model": settings.model,
                          "general_knowledge": summary(general), "project_evidence": summary(project)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
