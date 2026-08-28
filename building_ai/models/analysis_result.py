from dataclasses import dataclass, field
from typing import Any

from .semantic_result import SemanticResult


@dataclass(slots=True)
class AnalysisResult:
    semantic_results: list[SemanticResult] = field(default_factory=list)
    role_dict: dict[str, str] = field(default_factory=dict)
    ai_roles: dict[str, str] = field(default_factory=dict)
    slot_details: dict[str, Any] = field(default_factory=dict)
    unit_db: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def by_raw_name(self) -> dict[str, SemanticResult]:
        result: dict[str, SemanticResult] = {}
        for item in self.semantic_results:
            if item.raw_name in result:
                raise ValueError("raw_name is ambiguous; use by_point_id()")
            result[item.raw_name] = item
        return result

    def by_point_id(self) -> dict[str, SemanticResult]:
        return {item.point_id: item for item in self.semantic_results if item.point_id}
