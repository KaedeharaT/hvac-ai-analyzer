from building_ai.config.settings import DIRECT_PROMPT_VERSIONS
from building_ai.research.direct_prompt_templates import validate_direct_prompt_version


def test_all_direct_prompt_versions_retained():
    expected = {
        "direct_v1_name", "direct_v2_name_values",
        "direct_v3_name_values_stats", "direct_v4_full_context",
    }
    assert DIRECT_PROMPT_VERSIONS == expected
    assert all(validate_direct_prompt_version(item) == item for item in expected)
