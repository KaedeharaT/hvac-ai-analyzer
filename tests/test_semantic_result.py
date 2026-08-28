import pytest

from building_ai.models import SemanticResult, SemanticStatus


def test_status_adapter_preserves_abstention_and_review():
    assert SemanticResult("x", "other", abstained=True).status is SemanticStatus.ABSTAIN
    assert SemanticResult("x", "other", suspicious=True).status is SemanticStatus.REVIEW
    assert SemanticResult("x", "heat_source_flow").status is SemanticStatus.ACCEPT


def test_human_mapping_does_not_overwrite_ai_prediction():
    result = SemanticResult(
        "HP01 出水T", "heat_source_supply_temp",
        human_verified=True, human_label="heat_source_return_temp",
    )
    assert result.canonical_label == "heat_source_supply_temp"
    assert result.effective_label == "heat_source_return_temp"


def test_taxonomy_rejects_fabricated_label():
    with pytest.raises(ValueError):
        SemanticResult("x", "temperature_other")
