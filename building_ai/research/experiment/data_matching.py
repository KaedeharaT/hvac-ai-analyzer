"""GT 列名与单个 DataFrame 表头的确定性匹配工具。"""

import json
import re
import unicodedata
from difflib import SequenceMatcher


FUZZY_THRESHOLD = 0.90
AMBIGUITY_MARGIN = 0.03

_HYPHENS = str.maketrans({
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    "−": "-", "﹘": "-", "﹣": "-", "－": "-",
})


def normalize_column_name_for_matching(name: str) -> str:
    """只统一字符形式和空白，不删除设备编号或语义字符。"""
    text = unicodedata.normalize("NFKC", str(name))
    text = text.strip()
    while (
        len(text) >= 2
        and text[0] == text[-1]
        and text[0] in {"'", '"'}
    ):
        text = text[1:-1].strip()
    text = text.translate(_HYPHENS)
    text = text.replace("℃", "°C")
    text = re.sub(r"[º˚]\s*[cC]\b", "°C", text)
    text = re.sub(r"°\s*[cC]\b", "°C", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _candidate_record(raw, normalized, score):
    return {
        "column_raw": str(raw),
        "column_normalized": normalized,
        "score": float(score),
    }


def match_dataframe_column(
    gt_column,
    dataframe_columns,
    fuzzy_threshold=FUZZY_THRESHOLD,
    ambiguity_margin=AMBIGUITY_MARGIN,
):
    """在给定表头内按 exact → normalized_exact → fuzzy 匹配。"""
    gt_raw = str(gt_column)
    gt_normalized = normalize_column_name_for_matching(gt_raw)
    columns = [str(column) for column in dataframe_columns]

    exact = [column for column in columns if column == gt_raw]
    if len(exact) == 1:
        matched = exact[0]
        return {
            "gt_column_raw": gt_raw,
            "matched_column_raw": matched,
            "gt_column_normalized": gt_normalized,
            "matched_column_normalized": normalize_column_name_for_matching(matched),
            "match_method": "exact",
            "match_score": 1.0,
            "match_status": "matched",
            "match_candidates": json.dumps(
                [_candidate_record(matched, gt_normalized, 1.0)],
                ensure_ascii=False,
            ),
        }
    if len(exact) > 1:
        candidates = [
            _candidate_record(column, gt_normalized, 1.0)
            for column in exact
        ]
        return _unmatched_result(
            gt_raw, gt_normalized, "ambiguous", 1.0, candidates
        )

    normalized_pairs = [
        (column, normalize_column_name_for_matching(column))
        for column in columns
    ]
    normalized_exact = [
        (raw, normalized)
        for raw, normalized in normalized_pairs
        if normalized == gt_normalized
    ]
    if len(normalized_exact) == 1:
        matched, matched_normalized = normalized_exact[0]
        return {
            "gt_column_raw": gt_raw,
            "matched_column_raw": matched,
            "gt_column_normalized": gt_normalized,
            "matched_column_normalized": matched_normalized,
            "match_method": "normalized_exact",
            "match_score": 1.0,
            "match_status": "matched",
            "match_candidates": json.dumps(
                [_candidate_record(matched, matched_normalized, 1.0)],
                ensure_ascii=False,
            ),
        }
    if len(normalized_exact) > 1:
        candidates = [
            _candidate_record(raw, normalized, 1.0)
            for raw, normalized in normalized_exact
        ]
        return _unmatched_result(
            gt_raw, gt_normalized, "ambiguous", 1.0, candidates
        )

    ranked = sorted(
        [
            (
                SequenceMatcher(None, gt_normalized, normalized).ratio(),
                index,
                raw,
                normalized,
            )
            for index, (raw, normalized) in enumerate(normalized_pairs)
        ],
        key=lambda item: (-item[0], item[1]),
    )
    if not ranked:
        return _unmatched_result(gt_raw, gt_normalized, "missing", 0.0, [])

    best_score, _, best_raw, best_normalized = ranked[0]
    audit_candidates = [
        _candidate_record(raw, normalized, score)
        for score, _, raw, normalized in ranked[:5]
    ]
    if best_score < float(fuzzy_threshold):
        return _unmatched_result(
            gt_raw, gt_normalized, "missing", best_score, audit_candidates
        )

    second_score = ranked[1][0] if len(ranked) > 1 else None
    if (
        second_score is not None
        and best_score - second_score < float(ambiguity_margin)
    ):
        return _unmatched_result(
            gt_raw, gt_normalized, "ambiguous", best_score, audit_candidates
        )

    return {
        "gt_column_raw": gt_raw,
        "matched_column_raw": best_raw,
        "gt_column_normalized": gt_normalized,
        "matched_column_normalized": best_normalized,
        "match_method": "fuzzy",
        "match_score": float(best_score),
        "match_status": "matched",
        "match_candidates": json.dumps(audit_candidates, ensure_ascii=False),
    }


def _unmatched_result(gt_raw, gt_normalized, method, score, candidates):
    return {
        "gt_column_raw": gt_raw,
        "matched_column_raw": None,
        "gt_column_normalized": gt_normalized,
        "matched_column_normalized": None,
        "match_method": method,
        "match_score": float(score),
        "match_status": method,
        "match_candidates": json.dumps(candidates, ensure_ascii=False),
    }
