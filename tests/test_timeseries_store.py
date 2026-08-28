import pandas as pd

from building_ai.storage import TimeseriesStore


def test_mixed_bems_column_persists_and_reopens_without_arrow_failure(tmp_path):
    store = TimeseriesStore(tmp_path / "project_data")
    original = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=3, freq="h"),
        "Electrical Power": [10.0, "Alarm", 12.0],
        "Temperature": [7.0, 7.1, 7.2],
    })
    path = store.save("project-a", original)
    restored = store.load("project-a")
    assert path.exists()
    assert restored.shape == original.shape
    assert restored["Electrical Power"].tolist() == ["10.0", "Alarm", "12.0"]
