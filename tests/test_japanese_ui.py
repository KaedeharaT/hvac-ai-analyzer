"""Locale is presentation state, independent of scientific results and providers."""
import ast
import copy
import pickle
from dataclasses import asdict
import os
from pathlib import Path
from string import Formatter
from unittest.mock import Mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PyQt5.QtWidgets import QApplication
from building_ai.config import Settings
from building_ai.i18n import LanguageManager, tr
from building_ai.i18n.manager import TABLES
from building_ai.ui.main_window import MainWindow
from building_ai.ui.pages.energy_analysis_page import TimeSeriesChart
from scripts.capture_product_qa import _context


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_all_locales_have_identical_keys_placeholders_and_real_japanese():
    reference = TABLES['en_US']
    def fields(value):
        return sorted((name, spec, conversion) for _, name, spec, conversion in Formatter().parse(value) if name is not None)
    for locale, table in TABLES.items():
        assert set(table) == set(reference), locale
        for key, value in table.items():
            assert value.strip(), (locale, key)
            assert fields(value) == fields(reference[key]), (locale, key)
    # Proper names/symbols can remain unchanged; prose must be translated.
    unchanged = [key for key, value in reference.items() if len(value) > 15 and value == TABLES['ja_JP'][key]]
    assert not unchanged
    for path in (Path(__file__).parents[1] / 'building_ai' / 'ui').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'tr' and node.args and isinstance(node.args[0], ast.Constant):
                assert node.args[0].value in reference, (path.name, node.lineno, node.args[0].value)


def test_settings_dynamic_switch_preserves_scientific_and_user_state(app, tmp_path, monkeypatch):
    monkeypatch.setenv('BUILDING_AI_CONFIG_DIR', str(tmp_path / 'config'))
    context = _context(tmp_path)
    window = MainWindow(context); window.show(); app.processEvents()
    energy = window.page_by_key['energy_analysis']
    window.navigate_to('energy_analysis')
    energy.range.setCurrentIndex(energy.range.findData('custom'))
    energy.chart_filter.setCurrentIndex(energy.chart_filter.findData('temperature_trend'))
    equipment = window.page_by_key['equipment']
    equipment.refresh(); equipment.tabs.setCurrentIndex(3)
    knowledge = window.page_by_key['knowledge_base']
    knowledge.query.setText('COP'); knowledge.country_filter.setCurrentIndex(3)
    agent = window.page_by_key['agent']; agent.input.setPlainText('中文问题：请用中文回答。')
    raw = context.dataframe.copy(deep=True)
    diagnosis = pickle.dumps(asdict(context.diagnosis_result))
    semantic = copy.deepcopy(asdict(context.semantic_result))
    results = context.energy_analysis_view_result
    manager = context.llm_manager
    provider_settings = (context.settings.provider, context.settings.model, context.settings.api_base)
    monkeypatch.setattr(context, "reload_llm", Mock(side_effect=AssertionError("Locale must not reload provider")))
    analysis = Mock(side_effect=AssertionError('Locale change must not analyze'))
    monkeypatch.setattr(context.energy_analysis, 'analyze', analysis)
    before = (context.current_project.project_id, context.selected_equipment_id, context.selected_finding_id)
    for locale in ('zh_CN','en_US','ja_JP'):
        settings = window.page_by_key['settings']
        settings.language.setCurrentIndex(settings.language.findData(locale)); app.processEvents()
        assert LanguageManager.instance().language == locale
        assert Settings.load().language == locale
        assert tr('diagnostics_what_happened') in window.page_by_key['analysis'].finding_detail.toPlainText()
        assert (context.current_project.project_id, context.selected_equipment_id, context.selected_finding_id) == before
        assert context.energy_analysis_view_result is results
        assert context.llm_manager is manager
        assert (context.settings.provider, context.settings.model, context.settings.api_base) == provider_settings
        assert energy.range.currentData() == 'custom'
        assert energy.chart_filter.currentData() == 'temperature_trend'
        assert equipment.tabs.currentIndex() == 3
        assert knowledge.query.text() == 'COP' and knowledge.country_filter.currentData() == 'Japan'
        assert agent.input.toPlainText() == '中文问题：请用中文回答。'
        assert pickle.dumps(asdict(context.diagnosis_result)) == diagnosis
        assert asdict(context.semantic_result) == semantic
        assert context.dataframe.equals(raw)
    assert not analysis.called
    assert TimeSeriesChart._legend_label('AHP-01: Supply water temperature') == 'AHP-01: 往き温度'
    assert tr('energy_axis_temperature') == '温度 (°C)'
    window.close(); app.processEvents()
    reopened = MainWindow(__import__('building_ai.ui.context', fromlist=['ApplicationContext']).ApplicationContext(Settings.load()))
    assert reopened.page_by_key['settings'].language.currentData() == 'ja_JP'
    reopened.close()


def test_locale_does_not_change_calculated_energy(app, tmp_path, monkeypatch):
    monkeypatch.setenv('BUILDING_AI_CONFIG_DIR', str(tmp_path / 'config'))
    context = _context(tmp_path)
    reference = None
    for locale in TABLES:
        context.settings.language = locale; LanguageManager.instance().set_language(locale)
        result = context.energy_analysis.analyze(context.dataframe, context.semantic_result,
            context.current_project.project_id, context.import_metadata, context.equipment_organization,
            context.diagnosis_result.analytics)
        # Entire structured energy output, not only rounded display values.
        payload = result.to_dict()
        if reference is None: reference = payload
        assert payload == reference


def test_explicit_answer_language_overrides_ui(app):
    from building_ai.agent_controller import response_language
    LanguageManager.instance().set_language('ja_JP')
    assert response_language('HVACの保守点検について') == 'ja_JP'
    assert response_language('请用中文解释 COP。') == 'zh_CN'
    assert response_language('Please answer in English.') == 'en_US'
    assert response_language('中国語で回答してください。') == 'zh_CN'
    LanguageManager.instance().set_language('zh_CN')
    assert response_language('日本語で説明してください。') == 'ja_JP'
