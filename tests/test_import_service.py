import pandas as pd

from building_ai.services import ImportService


def test_csv_metadata(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=3, freq="15min"),
        "CH-1_LWT": [7.0, None, 7.2],
    }).to_csv(path, index=False)
    result = ImportService().load(path)
    assert result.metadata.rows == 3
    assert result.metadata.columns == 2
    assert result.metadata.time_column == "timestamp"
    assert result.metadata.sampling_interval == "0 days 00:15:00"


def test_excel_lists_and_selects_sheets(tmp_path):
    path = tmp_path / "sample.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"a": [1]}).to_excel(writer, sheet_name="A", index=False)
        pd.DataFrame({"b": [2]}).to_excel(writer, sheet_name="B", index=False)
    service = ImportService()
    assert service.list_sheets(path) == ["A", "B"]
    assert list(service.load(path, "B").dataframe.columns) == ["b"]


def test_date_and_time_columns_are_combined_and_all_sheets_are_supported(tmp_path):
    path = tmp_path / "mixed.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"Date": ["2025/01/01", "2025/01/01"], "Time": ["00:00", "00:15"], "Power kW": [1, 2]}).to_excel(writer, sheet_name="English", index=False)
        pd.DataFrame({"日時": ["2025年01月02日 00:00"], "電力 kW": [3]}).to_excel(writer, sheet_name="日本語", index=False)
    service = ImportService(); combined = service.load(path, "__all__")
    assert combined.metadata.time_column == "timestamp"
    assert combined.dataframe["timestamp"].notna().sum() == 2
    assert "__source_sheet__" in combined.dataframe
