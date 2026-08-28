import json
from pathlib import Path

import pandas as pd

from building_ai.services import SemanticService


FIXTURE = Path(__file__).parent / "fixtures" / "regression_v1"


def test_synthetic_multilingual_offline_fixture():
    frame = pd.read_csv(FIXTURE / "input.csv")
    expected = json.loads((FIXTURE / "expected_semantics.json").read_text(encoding="utf-8"))
    result = SemanticService().analyze_dataframe(frame)
    actual = result.by_raw_name()
    for name, label in expected["labels"].items():
        assert actual[name].canonical_label == label
    for name, status in expected["status"].items():
        assert actual[name].status.value == status
