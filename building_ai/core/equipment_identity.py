"""Stable, conservative equipment-identifier extraction for product code.

Research source: paper_research/src/bems_v4_egr/autonomous.py.  This module
only normalizes explicit identifiers; it never infers an identifier from value
correlation or similarly named equipment.
"""
from __future__ import annotations

import re
import unicodedata


_EQUIPMENT = re.compile(
    r"(?<![A-Z0-9])(?P<prefix>AHP|ASHP|WSHP|HP|CHILLER|CHU|CH|HEX|AHU|FCU|PCH|PWU|PUMP|"
    r"冷凍機|冷冻机|熱源|热源|ヒートポンプ|チラー|空調機|空调机|送風機|送风机|ポンプ)"
    r"(?:\s*(?:NO\.?|#))?[\s#_\-]*(?P<number>\d+(?:[\s#_\-]*\d+)*)(?:号機|號機)?(?![A-Z0-9])",
    re.IGNORECASE,
)


def normalize_equipment_id(value: str | None) -> str | None:
    """Return an explicit canonical ID such as ``HP-3-1``, or ``None``.

    NFKC makes full-width forms comparable, while the boundary-aware pattern
    keeps close numeric identifiers distinct.
    """
    text = unicodedata.normalize("NFKC", str(value or "")).upper()
    match = _EQUIPMENT.search(text)
    if not match:
        return None
    prefix = match.group("prefix").upper()
    number = re.sub(r"[\s#_-]+", "-", match.group("number")).strip("-")
    return f"{prefix}-{number}"
