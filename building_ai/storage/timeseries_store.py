from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd


class TimeseriesStore:
    """Project-owned parsed cache with legacy flat-file read compatibility."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, project_id: str, dataframe: pd.DataFrame) -> Path:
        target_dir = self.root / project_id / "normalized"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "timeseries.parquet"
        # BEMS workbooks frequently mix a numeric reading with a textual
        # status/error marker in one column.  Arrow rejects such Python object
        # columns.  Normalize only the storage representation: all-numeric
        # object columns remain numeric; truly mixed columns become nullable
        # strings and are still interpreted conservatively by the semantic and
        # quality layers after a restart.
        normalized = self._parquet_safe(dataframe)
        try:
            normalized.to_parquet(target, index=False)
            return target
        except (ImportError, ValueError, TypeError):
            target = target_dir / "timeseries.csv"
            normalized.to_csv(target, index=False)
            return target

    @staticmethod
    def _parquet_safe(dataframe: pd.DataFrame) -> pd.DataFrame:
        result = dataframe.copy()
        for name in result.columns:
            values = result[name]
            if not pd.api.types.is_object_dtype(values):
                continue
            non_null = values.dropna()
            if non_null.empty:
                result[name] = values.astype("string")
                continue
            numeric = pd.to_numeric(values, errors="coerce")
            if numeric.loc[non_null.index].notna().all():
                result[name] = numeric
            else:
                result[name] = values.astype("string")
        return result

    def load(self, project_id: str) -> pd.DataFrame:
        parquet = self.root / project_id / "normalized" / "timeseries.parquet"
        csv = self.root / project_id / "normalized" / "timeseries.csv"
        if parquet.exists():
            return pd.read_parquet(parquet)
        if csv.exists():
            return pd.read_csv(csv)
        # Projects created before managed storage used a flat data/timeseries cache.
        legacy_root = self.root.parent / "timeseries"
        parquet, csv = legacy_root / f"{project_id}.parquet", legacy_root / f"{project_id}.csv"
        if parquet.exists():
            return pd.read_parquet(parquet)
        if csv.exists():
            return pd.read_csv(csv)
        raise FileNotFoundError(f"No time-series data for project {project_id}")

    def exists(self, project_id: str) -> bool:
        try:
            self.load(project_id)
            return True
        except FileNotFoundError:
            return False

    def clear(self, project_id: str) -> None:
        target = (self.root / project_id / "normalized").resolve()
        project = (self.root / project_id).resolve()
        if target.parent == project and target.exists():
            shutil.rmtree(target)
