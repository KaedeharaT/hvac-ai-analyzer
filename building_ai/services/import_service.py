from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class ImportMetadata:
    source_file: str
    sheet: str | None
    available_sheets: list[str]
    rows: int
    columns: int
    time_column: str | None
    start: str | None
    end: str | None
    sampling_interval: str | None
    missing_ratio: float
    duplicate_rows: int
    dtypes: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ImportResult:
    dataframe: pd.DataFrame
    metadata: ImportMetadata


class ImportService:
    def list_sheets(self, path: str | Path) -> list[str]:
        source = Path(path)
        if source.suffix.lower() not in {".xlsx", ".xls"}:
            return []
        try:
            return list(pd.ExcelFile(source).sheet_names)
        except ImportError as exc:
            if source.suffix.lower() == ".xls":
                raise ValueError("XLS import requires the optional 'xlrd' dependency; convert to XLSX or install xlrd.") from exc
            raise
        except ValueError as exc:
            # Pandas gives a low-level engine message for legacy XLS files.
            if source.suffix.lower() == ".xls" and "xlrd" in str(exc).casefold():
                raise ValueError("XLS import requires the optional 'xlrd' dependency; convert to XLSX or install xlrd.") from exc
            raise

    def load(self, path: str | Path, sheet: str | int | None = None) -> ImportResult:
        source = Path(path)
        suffix = source.suffix.lower()
        available: list[str] = []
        selected: str | None = None
        if suffix == ".csv":
            # Research source: paper_research/src/bems_v2/ingest.py. BEMS
            # exports are commonly UTF-8 or Japanese legacy encodings.
            last_error: UnicodeDecodeError | None = None
            for encoding in ("utf-8-sig", "utf-8", "cp932", "shift_jis"):
                try:
                    df = pd.read_csv(source, encoding=encoding)
                    break
                except UnicodeDecodeError as exc:
                    last_error = exc
            else:
                raise ValueError(f"Cannot decode CSV {source}") from last_error
        elif suffix in {".xlsx", ".xls"}:
            available = self.list_sheets(source)
            if not available:
                raise ValueError("The workbook contains no worksheets")
            selected = available[0] if sheet is None else (available[sheet] if isinstance(sheet, int) else sheet)
            if selected != "__all__" and selected not in available:
                raise ValueError(f"Unknown worksheet: {selected}")
            if selected == "__all__":
                sheets = pd.read_excel(source, sheet_name=None)
                # Keep source provenance when combining sheets with different
                # schemas.  A regular column is used so the data remains portable.
                df = pd.concat([frame.assign(__source_sheet__=name) for name, frame in sheets.items()], ignore_index=True, sort=False)
            else:
                df = pd.read_excel(source, sheet_name=selected)
        else:
            raise ValueError("Only CSV and XLSX/XLS are supported")

        time_col, parsed = self._detect_time(df)
        interval = None
        start = end = None
        if parsed is not None and parsed.notna().any():
            valid = parsed.dropna().sort_values()
            start, end = valid.iloc[0].isoformat(), valid.iloc[-1].isoformat()
            diffs = valid.diff().dropna()
            if not diffs.empty:
                interval = str(diffs.median())
        metadata = ImportMetadata(
            source_file=str(source.resolve()), sheet=selected, available_sheets=available,
            rows=len(df), columns=len(df.columns), time_column=time_col,
            start=start, end=end, sampling_interval=interval,
            missing_ratio=float(df.isna().mean().mean()) if df.size else 0.0,
            duplicate_rows=int(df.duplicated().sum()),
            dtypes={str(k): str(v) for k, v in df.dtypes.items()},
        )
        return ImportResult(df, metadata)

    @staticmethod
    def _parse_datetime(values: pd.Series) -> pd.Series:
        text = values.astype(str).str.strip().str.replace("年", "-", regex=False).str.replace("月", "-", regex=False).str.replace("日", " ", regex=False).str.replace("時", ":", regex=False).str.replace("分", "", regex=False)
        return pd.to_datetime(text.where(values.notna()), errors="coerce")

    @classmethod
    def _detect_time(cls, df: pd.DataFrame) -> tuple[str | None, pd.Series | None]:
        hints = ("time", "date", "datetime", "timestamp", "時刻", "日時", "时间", "日期")
        candidates = [
            c for c in df.columns if any(h in str(c).lower() for h in hints)
        ] + list(df.columns[:3])
        seen: set[str] = set()
        for col in candidates:
            if str(col) in seen:
                continue
            seen.add(str(col))
            # A date-only field must not win over an adjacent clock field.
            # Preserve the combined timestamp so real sampling intervals remain
            # available to downstream energy integration.
            if any(h in str(col).lower() for h in ("date", "日付", "日期")) and any(
                other != col and any(h in str(other).lower() for h in ("time", "時刻", "时间")) for other in df.columns
            ):
                continue
            if any(h in str(col).lower() for h in ("time", "時刻", "时间")) and any(
                other != col and any(h in str(other).lower() for h in ("date", "日付", "日期")) for other in df.columns
            ):
                continue
            parsed = cls._parse_datetime(df[col])
            if len(parsed) and parsed.notna().mean() >= 0.8:
                return str(col), parsed
        date_cols = [c for c in df.columns if any(h in str(c).lower() for h in ("date", "日付", "日期"))]
        clock_cols = [c for c in df.columns if any(h in str(c).lower() for h in ("time", "時刻", "时间"))]
        for date_col in date_cols:
            for clock_col in clock_cols:
                if date_col == clock_col:
                    continue
                parsed = cls._parse_datetime(df[date_col].astype(str) + " " + df[clock_col].astype(str))
                if len(parsed) and parsed.notna().mean() >= 0.5:
                    combined_name = "timestamp"
                    number = 1
                    while combined_name in df.columns:
                        number += 1; combined_name = f"timestamp_{number}"
                    df[combined_name] = parsed
                    return combined_name, parsed
        return None, None
