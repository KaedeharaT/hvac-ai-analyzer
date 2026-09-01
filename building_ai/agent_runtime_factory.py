"""Select the backward-compatible Single or optional Multi-Agent runtime."""
from __future__ import annotations

from typing import Any, Literal

from building_ai.agent_runtime import AgentRuntime
from building_ai.multi_agent_runtime import MultiAgentRuntime


AgentMode = Literal["single", "multi"]


def create_agent_runtime(context: Any, agent_mode: str | None = None) -> AgentRuntime | MultiAgentRuntime:
    mode = agent_mode or getattr(getattr(context, "settings", None), "agent_mode", "single")
    if mode not in {"single", "multi"}:
        raise ValueError("agent_mode must be 'single' or 'multi'")
    return MultiAgentRuntime(context) if mode == "multi" else AgentRuntime(context)
