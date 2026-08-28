from building_ai.research.experiment.data_matching import normalize_column_name_for_matching


def test_matching_normalization_is_retained():
    assert normalize_column_name_for_matching(" ＣＨ－１＿ＬＷＴ ") == normalize_column_name_for_matching("CH-1_LWT")
