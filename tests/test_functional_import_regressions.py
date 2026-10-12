"""Synthetic regressions for defects observed in native multi-project QA."""
from datetime import time
import pandas as pd
import pytest

from building_ai.config import Settings
from building_ai.services.import_service import ImportService
from building_ai.services.semantic_service import SemanticService
from building_ai.ui.context import ApplicationContext


def test_clock_only_never_invents_a_calendar_date(tmp_path):
    path = tmp_path / 'clocks.xlsx'
    pd.DataFrame({'時刻': [time(0), time(0, 1)], 'power kW': [1, 2]}).to_excel(path, index=False)
    result = ImportService().load(path)
    assert result.metadata.start is None and result.metadata.time_column is None
    assert result.metadata.temporal_columns == ['時刻']
    assert 'import_date_required' in result.metadata.warnings


def test_unnamed_date_and_clock_preserve_minute_sampling(tmp_path):
    path = tmp_path / 'split.xlsx'
    pd.DataFrame({0: ['2025-01-01'] * 3, 1: ['00:00:00', '00:01:00', '00:02:00'], 'power kW': [1, 2, 3]}).to_excel(path, index=False)
    result = ImportService().load(path)
    assert result.metadata.sampling_interval == '0 days 00:01:00'
    assert result.metadata.end == '2025-01-01T00:02:00'
    assert set(result.metadata.temporal_columns) == {'0', '1', 'timestamp'}


def test_excel_metadata_units_are_evidence_not_readings(tmp_path):
    path = tmp_path / 'metadata.xlsx'
    pd.DataFrame({'time': [None, None, '2025-01-01', '2025-01-02'],
                  'CH-1 Power': ['計測', 'W', 1000, 2000]}).to_excel(path, index=False)
    result = ImportService().load(path)
    assert result.metadata.rows == 2
    assert result.metadata.metadata_rows == 2
    semantic = SemanticService().analyze_dataframe(result.dataframe, column_units=result.metadata.column_units, require_unit_evidence=True)
    assert semantic.by_raw_name()['CH-1 Power'].unit == 'W'


def test_preamble_japanese_legacy_csv_and_duplicate_names(tmp_path):
    path = tmp_path / 'preamble.csv'
    path.write_text('SYNTHETIC TEST DATA\n時刻;電力 kW;電力 kW\n2025/01/01 00:00;1;2\n2025/01/01 01:00;2;3\n', encoding='cp932')
    result = ImportService().load(path)
    assert result.metadata.header_row == 2
    assert result.metadata.rows == 2
    assert len(set(result.dataframe.columns)) == 3
    assert result.metadata.time_column == '時刻'


def test_normalized_header_collision_keeps_both_point_identities(tmp_path, monkeypatch):
    monkeypatch.setenv('BUILDING_AI_CONFIG_DIR', str(tmp_path))
    frame = pd.DataFrame({'CH-1 LWT ℃': [6, 7], 'CH-1 LWT °C': [8, 9]})
    result = SemanticService().analyze_dataframe(frame, project_id='collision')
    assert {item.raw_name for item in result.semantic_results} == set(frame.columns)
    assert len(result.by_point_id()) == 2


def test_product_unit_confirmation_and_project_reset_persist(tmp_path, monkeypatch):
    monkeypatch.setenv('BUILDING_AI_CONFIG_DIR', str(tmp_path / 'config'))
    settings = Settings(data_dir=tmp_path / 'data')
    context = ApplicationContext(settings)
    first = context.projects.create('Synthetic A'); context.open_project(first.project_id)
    path = tmp_path / 'power.csv'
    pd.DataFrame({'time': pd.date_range('2025-01-01', periods=2, freq='h'), 'CH-1 Power': [1000, 2000]}).to_csv(path, index=False)
    context.import_data(str(path)); context.run_semantics()
    point = context.semantic_result.by_raw_name()['CH-1 Power']
    assert point.unit is None and point.needs_review
    assert context.energy_analysis_result.summary['total_energy_kwh'] is None
    context.save_review(point.point_id, 'heat_source_power', 'fixture unit documented as W', 'CH-1', unit='W')
    context.selected_period = '7d'
    second = context.projects.create('Synthetic B'); context.open_project(second.project_id)
    assert context.dataframe is None and context.semantic_result is None and not context.equipment
    assert context.energy_analysis_result is None and context.selected_period == 'all'
    reopened = ApplicationContext(settings); reopened.open_project(first.project_id)
    point = reopened.semantic_result.by_raw_name()['CH-1 Power']
    assert point.unit == 'W' and point.confirmed_label == 'heat_source_power'
    reopened.run_semantics()
    assert reopened.semantic_result.by_raw_name()['CH-1 Power'].unit == 'W'
    assert reopened.semantic_result.by_raw_name()['CH-1 Power'].confirmed_equipment_id == 'CH-1'
    persisted = reopened.projects.load_semantics(first.project_id)[0]
    assert persisted.debug_metadata['imported_unit_evidence']['source'] == 'human_review'


def test_confirmed_abstained_signal_is_available_for_equipment_binding():
    from building_ai.models import SemanticResult
    from building_ai.services.equipment_service import EquipmentService
    point = SemanticResult('synthetic unnamed sensor', 'other', abstained=True, gate_status='ABSTAIN',
                           confirmed_label='heat_source_supply_temp', confirmed_equipment_id='CH-1', unit='°C')
    organization = EquipmentService().organize('test', [point])
    assert len(organization.equipment) == 1
    assert 'heat_source_supply_temp' in organization.heat_sources[0].points_by_role


def test_energy_only_equipment_is_not_a_heat_source_validation_failure():
    from building_ai.models import AnalysisResult, SemanticResult
    from building_ai.services.analytics_service import AnalyticsResult
    from building_ai.services.energy_analysis_service import EnergyAnalysisService
    from building_ai.services.consistency_validation import validate_analysis
    frame = pd.DataFrame({'time': pd.date_range('2025-01-01', periods=3, freq='h'),
                          'PCH-1 kWh': [1, 2, 3], 'PCH-2 kWh': [4, 5, 6]})
    points = [SemanticResult(name, 'other', point_id=name, equipment_id=equipment, physical_quantity='energy', unit='kWh')
              for name, equipment in [('PCH-1 kWh', 'PCH-1'), ('PCH-2 kWh', 'PCH-2')]]
    energy = EnergyAnalysisService().analyze(frame, AnalysisResult(points, {}), 'test', {'time_column': 'time'})
    assert energy.charts['equipment_ranking']['data']
    analytics = AnalyticsResult([])
    assert validate_analysis(analytics, [], [], energy).passed
    energy.charts['equipment_ranking']['data'][0]['name'] = 'unrelated equipment'
    assert not validate_analysis(analytics, [], [], energy).passed
