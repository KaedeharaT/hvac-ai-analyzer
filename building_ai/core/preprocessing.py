from __future__ import annotations

import unicodedata
import re

import pandas as pd


def normalize_header(value: object) -> str:
    """Normalize Unicode and presentation variants without changing identity."""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("－", "-").replace("（", "(").replace("）", ")")
    return re.sub(r"\s+", " ", text).strip()


def normalize_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    result.columns = [normalize_header(c) for c in result.columns]
    return result


def find_time_column(dataframe: pd.DataFrame) -> str | None:
    """Identify a real timestamp column before point-wise semantic inference."""
    tokens = ("time", "timestamp", "date", "datetime", "日時", "時刻", "日付", "时间", "日期")
    for column in dataframe.columns:
        if not any(token in str(column).lower() for token in tokens):
            continue
        parsed = pd.to_datetime(dataframe[column], errors="coerce")
        if len(parsed) and parsed.notna().mean() >= 0.6:
            return str(column)
    return None
