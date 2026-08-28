from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(slots=True)
class ToolResult:
    tool: str
    ok: bool
    data: Any = None
    error: str | None = None


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, function: Callable[..., Any]) -> None:
        self._tools[name] = function

    def names(self) -> list[str]:
        return sorted(self._tools)

    def call(self, name: str, **kwargs: Any) -> ToolResult:
        if name not in self._tools:
            return ToolResult(name, False, error="Unknown tool")
        try:
            return ToolResult(name, True, self._tools[name](**kwargs))
        except Exception as exc:
            return ToolResult(name, False, error=str(exc))
