from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
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
    warnings: list[str] = field(default_factory=list)
    column_units: dict[str, str] = field(default_factory=dict)
    temporal_columns: list[str] = field(default_factory=list)
    header_row: int = 1
    metadata_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ImportResult:
    dataframe: pd.DataFrame
    metadata: ImportMetadata


class ImportService:
    TIME_HINTS = ("timestamp", "datetime", "time", "date", "日時", "時刻", "日付", "时间", "日期")
    UNITS = {"kw": "kW", "w": "W", "mw": "MW", "kwh": "kWh", "wh": "Wh", "mwh": "MWh",
             "°c": "°C", "℃": "°C", "m3/h": "m3/h", "m³/h": "m3/h", "l/min": "L/min",
             "l/s": "L/s", "gj": "GJ", "mj": "MJ", "gj/h": "GJ/h", "%": "%",
             "pa": "Pa", "kpa": "kPa", "mpa": "MPa", "hz": "Hz", "a": "A", "v": "V"}

    @classmethod
    def _header_offset(cls, rows: list[list]) -> int:
        """Recognize a labelled table following a short explanatory preamble."""
        for index, row in enumerate(rows[:30]):
            cells = [str(value).strip().lower() for value in row if pd.notna(value)]
            if len(cells) >= 2 and any(cell in cls.TIME_HINTS for cell in cells):
                return index
            if index == 0 and len(cells) >= 2:
                # An ordinary header without a time label must remain intact.
                return 0
        return 0

    @classmethod
    def _strip_metadata_rows(cls, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], int]:
        units: dict[str, str] = {}
        removed = 0
        labels = {"計測", "測定", "measurement", "measured", "unit", "units", "単位", "单位"}
        for _, row in frame.head(5).iterrows():
            values = {str(column): str(value).strip().lower() for column, value in row.items() if pd.notna(value)}
            if not values or not all(value in cls.UNITS or value in labels for value in values.values()):
                break
            units.update({column: cls.UNITS[value] for column, value in values.items() if value in cls.UNITS})
            removed += 1
        return frame.iloc[removed:].reset_index(drop=True), units, removed

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
        header = 0
        warnings: list[str] = []
        if suffix == ".csv":
            # Research source: paper_research/src/bems_v2/ingest.py. BEMS
            # exports are commonly UTF-8 or Japanese legacy encodings.
            last_error: UnicodeDecodeError | None = None
            with source.open("rb") as stream:
                prefix = stream.read(4)
            encodings = ("utf-16",) if prefix.startswith((b"\xff\xfe", b"\xfe\xff")) else ("utf-8-sig", "cp932", "gb18030", "shift_jis")
            for encoding in encodings:
                try:
                    with source.open(encoding=encoding) as stream:
                        sample = [stream.readline() for _ in range(30)]
                    # Delimiter evidence comes from table rows, not a title line.
                    delimiter = max((",", ";", "\t"), key=lambda sep: sum(line.count(sep) for line in sample))
                    header = self._header_offset(list(csv.reader(sample, delimiter=delimiter)))
                    df = pd.read_csv(source, encoding=encoding, sep=delimiter, header=header)
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
                preview = pd.read_excel(source, sheet_name=selected, header=None, nrows=30)
                header = self._header_offset(preview.values.tolist())
                df = pd.read_excel(source, sheet_name=selected, header=header)
        else:
            raise ValueError("Only CSV and XLSX/XLS are supported")

        df.columns = [str(column) for column in df.columns]
        df, column_units, metadata_rows = self._strip_metadata_rows(df)
        if header or metadata_rows:
            warnings.append("import_metadata_rows_removed")
        time_col, parsed = self._detect_time(df)
        temporal_columns = [str(c) for c in df.columns if self._clock_values(df[c])]
        temporal_columns.extend(df.attrs.get("building_ai_time_source_columns", []))
        if time_col:
            temporal_columns.append(time_col)
            # Persist exactly the parsed dates so every consumer and restart
            # uses the same timestamp, including Japanese date notation.
            df[time_col] = parsed
        else:
            warnings.append("import_date_required" if temporal_columns else "import_time_unavailable")
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
            warnings=warnings, column_units=column_units, temporal_columns=list(dict.fromkeys(temporal_columns)),
            header_row=header + 1, metadata_rows=metadata_rows,
        )
        return ImportResult(df, metadata)

    @staticmethod
    def _parse_datetime(values: pd.Series) -> pd.Series:
        text = values.astype(str).str.strip().str.replace("年", "-", regex=False).str.replace("月", "-", regex=False).str.replace("日", " ", regex=False).str.replace("時", ":", regex=False).str.replace("分", "", regex=False)
        # A clock without a calendar date must never acquire today's date.
        dated = text.str.contains(r"(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}|^\d{8}$)", regex=True)
        return pd.to_datetime(text.where(values.notna() & dated), errors="coerce")

    @staticmethod
    def _clock_values(values: pd.Series) -> bool:
        observed = values.dropna().astype(str).str.strip()
        return bool(len(observed) and observed.str.match(r"^\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?$").mean() >= .8)

    @classmethod
    def _detect_time(cls, df: pd.DataFrame) -> tuple[str | None, pd.Series | None]:
        hints = cls.TIME_HINTS
        candidates = [
            c for c in df.columns if any(h in str(c).lower() for h in hints)
        ] + list(df.columns[:3])
        # Some logger exports use two unnamed columns for date and clock.
        # Require content evidence for both and combine before choosing date.
        clock_cols = [c for c in df.columns if cls._clock_values(df[c])]
        date_cols = []
        for col in dict.fromkeys(candidates):
            values = cls._parse_datetime(df[col])
            valid = values.dropna()
            observed = int(df[col].notna().sum())
            if observed and len(valid) / observed >= .8 and (valid.dt.normalize() == valid).all():
                date_cols.append(col)
        for date_col in date_cols:
            for clock_col in clock_cols:
                parsed = cls._parse_datetime(cls._parse_datetime(df[date_col]).dt.strftime("%Y-%m-%d") + " " + df[clock_col].astype(str))
                if parsed.notna().any():
                    combined_name = "timestamp"
                    number = 1
                    while combined_name in df.columns:
                        number += 1; combined_name = f"timestamp_{number}"
                    df[combined_name] = parsed
                    df.attrs["building_ai_time_source_columns"] = [str(date_col), str(clock_col)]
                    return combined_name, parsed
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
            if df[col].notna().any() and parsed.notna().sum() / df[col].notna().sum() >= 0.8:
                return str(col), parsed
        return None, None
