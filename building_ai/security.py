"""Security boundaries for the bounded, read-only agent runtime."""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence


_INJECTION_PATTERNS = (
    r"\bignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+instructions?\b",
    r"\bdisregard\s+(?:all\s+)?(?:previous|prior|above)\b",
    r"\b(?:reveal|show|print|dump|exfiltrate)\s+(?:the\s+)?(?:system\s+)?prompt\b",
    r"\byou\s+are\s+now\b",
    r"\bact\s+as\s+(?:a\s+)?(?:system|developer)\b",
    r"\bignore\b.{0,40}\b(?:rules?|instructions?)\b",
    r"\b(?:show|reveal|print|dump)\b.{0,40}\bsystem\s+prompt\b",
    r"忽略(?:以前|之前|所有)?(?:的)?(?:指令|规则)",
    r"(?:显示|透露|导出).{0,30}(?:系统提示词|system prompt)",
)
_INJECTION = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def contains_prompt_injection(value: object) -> bool:
    """Detect instruction-override attempts at an untrusted input boundary."""
    return bool(_INJECTION.search(str(value)))


def find_untrusted_instruction_paths(value: object, path: str = "$") -> list[str]:
    """Return locations of instruction-like text without copying it into logs."""
    if isinstance(value, Mapping):
        return [item for key, child in value.items() for item in find_untrusted_instruction_paths(child, f"{path}.{key}")]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [item for index, child in enumerate(value) for item in find_untrusted_instruction_paths(child, f"{path}[{index}]")]
    return [path] if isinstance(value, str) and contains_prompt_injection(value) else []
