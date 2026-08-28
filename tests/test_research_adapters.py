import json
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from building_ai.research.client_adapter import ResearchLLMClientAdapter
from building_ai.research.hvac_power_col_memory import normalize_llm_slots
from building_ai.research.unit_adapter import extract_research_unit_db


FIXTURE = Path(__file__).parent / "fixtures" / "regression_v1"


def test_research_client_exposes_openai_shape():
    client = Mock()
    client.chat_text.return_value = "heat_source_power"
    adapter = ResearchLLMClientAdapter(client)
    response = adapter.chat.completions.create(
        messages=[{"role": "system", "content": "system"},
                  {"role": "user", "content": "classify"}],
        temperature=0, seed=0,
    )
    assert response.choices[0].message.content == "heat_source_power"
    client.chat_text.assert_called_once()


def test_frozen_c1_c8_response_normalizes():
    frozen = json.loads((FIXTURE / "frozen_llm_responses.json").read_text(encoding="utf-8"))
    normalized = normalize_llm_slots(frozen["CH-1_LWT"])
    assert set(normalized) == {f"C{i}" for i in range(1, 9)}
    assert normalized["C6"]["unit"] == "°C"


def test_research_units_come_from_c6_not_product_engine():
    slots = {f"C{i}": {} for i in range(1, 9)}
    slots["C6"] = {"unit": "legacy-unit", "unit_type": "temp", "confidence": 0.91}
    with patch("building_ai.research.unit_adapter.llm_score_all_slots", return_value=slots):
        units = extract_research_unit_db(pd.DataFrame({"x": [1, 2]}))
    assert units["x"] == {"unit": "legacy-unit", "unit_type": "temp",
                           "confidence": 0.91, "source": "research_c6"}
