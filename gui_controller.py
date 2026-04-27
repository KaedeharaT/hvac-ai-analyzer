import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QComboBox, QLineEdit, QPushButton, QCheckBox,
                             QGroupBox, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt
from analysis_core import analyze_data  # 从核心模块导入分析函数
from PyQt5 import QtGui
from PyQt5.QtWidgets import QTextEdit
import pandas as pd
import traceback
import pprint
from analysis_core import safe_fmt
from hvac_power_col_memory import batch_physical_role_review
from analysis_core import ensure_power_columns
from analysis_core import client as core_client
import os, datetime

def safe_get_stats(unit_result):
    return unit_result.get('stats') \
        or unit_result.get('cooling_stats') \
        or unit_result.get('heating_stats') \
        or {}

class HVACAnalyzerGUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("空調システムデータ自動分析ツール")
        self.setGeometry(100, 100, 1200, 700)  # 放大窗口

        # 总体横向布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.outer_layout = QHBoxLayout()
        self.central_widget.setLayout(self.outer_layout)

        # 左侧参数配置区
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout()
        self.left_panel.setLayout(self.left_layout)
        self.outer_layout.addWidget(self.left_panel, 1)

        # 原有各setup_xxx全部都加到左侧
        self.setup_file_selection()
        self.setup_load_analysis()
        self.setup_grouping_config()
        self.setup_cop_analysis()
        self.setup_analysis_options()
        self.setup_visualization_config()
        self.setup_action_buttons()

        # 右侧：仅结果显示，拉大
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout()
        self.right_panel.setLayout(self.right_layout)
        self.outer_layout.addWidget(self.right_panel, 2)

        self.setup_result_display()  # 这会加到 right_layout

        self.statusBar().showMessage("準備完了")
        self.current_results = {}
        self.openai_client = core_client
    def setup_cop_analysis(self):
        """COP分析配置区域"""
        group = QGroupBox("COP分析設定")
        layout = QVBoxLayout()

        self.enable_cop_check = QCheckBox("COP分析を有効にする")
        self.enable_cop_check.setChecked(True)

        self.water_shc_edit = QLineEdit()
        self.water_shc_edit.setPlaceholderText("水の比熱容量 (デフォルト: 4.18 kJ/kg·K)")
        self.water_shc_edit.setText("4.18")

        layout.addWidget(self.enable_cop_check)
        layout.addWidget(QLabel("水の比熱容量 (kJ/kg·K):"))
        layout.addWidget(self.water_shc_edit)

        group.setLayout(layout)
        self.left_layout.insertWidget(4, group)

    def setup_analysis_options(self):
        """分析选项区域（新加）"""
        group = QGroupBox("分析オプション")
        layout = QVBoxLayout()

        self.analyze_columns_check = QCheckBox("列名自動分類（AI/ルール）を有効にする")
        self.analyze_columns_check.setChecked(True)

        layout.addWidget(self.analyze_columns_check)

        self.export_slots_check = QCheckBox("导出C1–C8详细分数（CSV）")  # 新增
        self.export_slots_check.setChecked(False)  # 默认关闭
        layout.addWidget(self.export_slots_check)

        group.setLayout(layout)
        self.left_layout.addWidget(group)

    def setup_file_selection(self):
        """文件选择区域"""
        group = QGroupBox("データファイル選択")
        layout = QHBoxLayout()

        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("xlsxまたはCSVファイルを選択...")

        browse_btn = QPushButton("参照...")
        browse_btn.clicked.connect(self.browse_file)

        layout.addWidget(self.file_path_edit, 4)
        layout.addWidget(browse_btn, 1)
        group.setLayout(layout)
        self.left_layout.addWidget(group)

    def browse_file(self):
        """打开文件对话框"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "データファイルを選択", "",
            "データファイル (*.xlsx *.xls *.csv);;すべてのファイル (*)"
        )
        if file_path:
            self.file_path_edit.setText(file_path)

    def setup_grouping_config(self):
        """分组配置区域"""
        group = QGroupBox("グループ設定")
        layout = QVBoxLayout()

        # 分组模式选择
        self.group_mode = QComboBox()
        self.group_mode.addItems([
            #"設備ごとに分類する",
            #"データ種類ごとに分類する",
            #"設備→データ種類の順）",
            "AIに任せ"
        ])

        # 自定义规则输入
        self.custom_rule_edit = QLineEdit()
        self.custom_rule_edit.setPlaceholderText("未完成、入力する内容を理解できるように")

        # 机组识别设置
        self.unit_pattern_edit = QLineEdit()
        self.unit_pattern_edit.setText("(unit|ch|device|ユニット|番号|no)[_\-]?(\d+)")

        # 添加AI分类开关
        self.use_ai_check = QCheckBox("AI補助分類を有効にする")
        self.use_ai_check.setChecked(True)  # 默认启用

        layout.addWidget(QLabel("分類方式:"))
        layout.addWidget(self.group_mode)
        layout.addWidget(QLabel("ユーザー設定分類方式:"))
        layout.addWidget(self.custom_rule_edit)
        #layout.addWidget(QLabel("ユニット番号識別パターン:"))
        #layout.addWidget(self.unit_pattern_edit)
        layout.addWidget(self.use_ai_check)

        group.setLayout(layout)
        self.left_layout.addWidget(group)

    def setup_load_analysis(self):
        """负荷分析配置区域"""
        group = QGroupBox("負荷率分析設定")
        layout = QVBoxLayout()

        # 制冷量输入
        self.cooling_capacity_edit = QLineEdit()
        self.cooling_capacity_edit.setPlaceholderText("システムの総冷却能力(kW)を入力...")
        self.cooling_capacity_edit.setText("1000")

        # 启用负荷分析复选框
        self.enable_load_check = QCheckBox("負荷率分析を有効にする")
        self.enable_load_check.setChecked(True)


        # 气象数据配置
        weather_group = QGroupBox("外気温度データ")
        weather_layout = QVBoxLayout()

        self.enable_weather_check = QCheckBox("気象データを取得する")
        self.enable_weather_check.setChecked(True)

        # 位置输入
        loc_layout = QHBoxLayout()
        loc_layout.addWidget(QLabel("緯度:"))
        self.lat_edit = QLineEdit()
        self.lat_edit.setPlaceholderText("33.6")
        self.lat_edit.setText("33.6")
        loc_layout.addWidget(self.lat_edit)

        loc_layout.addWidget(QLabel("経度:"))
        self.lng_edit = QLineEdit()
        self.lng_edit.setPlaceholderText("130.4167")
        self.lng_edit.setText("130.4167")
        loc_layout.addWidget(self.lng_edit)

        # API密钥
        #self.weather_api_edit = QLineEdit()
        #self.weather_api_edit.setPlaceholderText("Open-Meteo APIキー ")

        weather_layout.addWidget(self.enable_weather_check)
        weather_layout.addLayout(loc_layout)
        #weather_layout.addWidget(QLabel("APIキー:"))
        #weather_layout.addWidget(self.weather_api_edit)
        weather_group.setLayout(weather_layout)

        layout.addWidget(QLabel("冷却能力 (kW):"))
        layout.addWidget(self.cooling_capacity_edit)
        layout.addWidget(self.enable_load_check)
        layout.addWidget(weather_group)

        group.setLayout(layout)
        self.left_layout.insertWidget(3, group)
    def setup_visualization_config(self):
        """可视化配置区域"""
        group = QGroupBox("グラフ生成")
        layout = QVBoxLayout()

        # 图表类型选择
        self.chart_type = QComboBox()
        self.chart_type.addItems([
            "折れ線グラフ",
            "棒グラフ",
            "散布図",

        ])

        # 参数选择
        self.param_group = QGroupBox("グラフにするデータを選択")
        param_layout = QVBoxLayout()

        self.temp_check = QCheckBox("温度")
        self.flow_check = QCheckBox("流量")
        self.power_check = QCheckBox("消費電力")
        self.other_edit = QLineEdit()
        self.other_edit.setPlaceholderText("他のパラメータキーワードを入力")

        param_layout.addWidget(self.temp_check)
        param_layout.addWidget(self.flow_check)
        param_layout.addWidget(self.power_check)
        param_layout.addWidget(self.other_edit)
        self.param_group.setLayout(param_layout)

        layout.addWidget(QLabel("グラフタイプ:"))
        layout.addWidget(self.chart_type)
        layout.addWidget(self.param_group)

        group.setLayout(layout)
        self.left_layout.addWidget(group)

    def setup_result_display(self):
        """初始化结果显示区域"""
        result_group = QGroupBox("分析結果")
        layout = QVBoxLayout()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        # 自动换行，避免横向滚动条
        self.result_text.setLineWrapMode(QTextEdit.WidgetWidth)
        self.result_text.setStyleSheet("font-family: Consolas; font-size: 13pt; min-height:600px;")
        layout.addWidget(self.result_text)
        result_group.setLayout(layout)
        self.right_layout.addWidget(result_group)

    def setup_action_buttons(self):
        """操作按钮区域"""
        layout = QHBoxLayout()

        analyze_btn = QPushButton("分析開始")
        analyze_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        analyze_btn.clicked.connect(self.run_analysis)


        layout.addWidget(analyze_btn)

        self.left_layout.addLayout(layout)

    def display_results(self):
        """显示分析结果到GUI界面"""
        if not self.current_results:
            return

        # 清空现有内容
        self.result_text.clear()

        # 显示基本结果信息
        self.result_text.append("=== 分析結果 ===\n")

        # ==== 【新增：AI自动单位判定详细显示】====
        unit_db = self.current_results.get('unit_db', None)
        if unit_db:
            self.result_text.append("\n=== 各字段AI推断单位 ===")
            for col, info in unit_db.items():
                unit_str = f"单位: {info.get('unit', '-')}"
                reason_str = f"理由: {info.get('reason', '-')}"
                convert_tip = f"换算建议: {info.get('convert_tip', '-')}"
                self.result_text.append(
                    f"・{col}\n   {unit_str}\n   {reason_str}\n   {convert_tip}\n"
                )
        # ==== 【新增END】====



        # ==== 【新增：单位推断+置信度结果显示】====
        # 支持单组与多组（多机组字典，单组直接list）
        unit_combo_scores = self.current_results.get('unit_combo_scores', None)
        if unit_combo_scores:
            self.result_text.append("\n=== 本次物理量单位自动推断 ===")
            # 多机组情况：unit_combo_scores = { "机组A": [...], "机组B": [...] }
            if isinstance(unit_combo_scores, dict):
                for group, combos in unit_combo_scores.items():
                    self.result_text.append(f"\n【{group}】")
                    if not combos:
                        self.result_text.append("  未获取到单位推断结果")
                        continue
                    for combo in combos:
                        if not isinstance(combo, dict):
                            self.result_text.append(
                                f"[警告] 单位组合结果不是dict，而是{type(combo)}：{combo}"
                            )
                            continue  # 跳过这条
                        self.result_text.append(
                            f"・单位组合: {combo.get('unit_combo', '')} | "
                            f"COP={safe_fmt(combo.get('cop'), '.2f')} | "
                            f"置信度: {safe_fmt(combo.get('confidence'), '.2f')} | "
                            f"推断理由: {combo.get('reason', '-')}"
                        )

                    # 排序之前，也要过滤掉非dict项
                    combos_dict = [x for x in combos if isinstance(x, dict)]
                    sorted_combos = sorted(combos_dict, key=lambda x: x.get('confidence', 0), reverse=True)

                    if sorted_combos:
                        best = sorted_combos[0]
                        self.result_text.append(
                            f"→ 本次分析采用单位: {best.get('unit_combo')}"
                        )
                    else:
                        self.result_text.append(
                            "  未能推断出可信的单位组合"
                        )
            # 单组情况
            elif isinstance(unit_combo_scores, list):
                flat = []
                for item in unit_combo_scores:
                    # 兼容旧结构
                    combo = item.get('result') if isinstance(item, dict) and 'result' in item else item
                    if isinstance(combo, dict):
                        flat.append(combo)
                for combo in flat:
                    self.result_text.append(
                        f"・单位组合: {combo.get('unit_combo', '')} | "
                        f"COP均值: {safe_fmt(combo.get('cop_mean'), '.2f')} | "
                        f"样本数: {safe_fmt(combo.get('n_valid'), '.0f')} | "
                        f"置信度: {safe_fmt(combo.get('confidence'), '.2f')} | "
                        f"推断理由: {combo.get('reason', '-')}"
                    )
                sorted_combos = sorted(flat, key=lambda x: x.get('confidence', 0), reverse=True)
                if sorted_combos:
                    best = sorted_combos[0]
                    self.result_text.append(
                        f"→ 本次分析采用单位: {best.get('unit_combo')} (置信度最高 {safe_fmt(best.get('confidence'), '.2f')})")

            self.result_text.append("\n")
        # ==== 【新增END】====

        # 显示负荷分析结果 (新增)
        load_analysis = self.current_results.get('load_analysis')
        if load_analysis and isinstance(load_analysis, dict):
            self.result_text.append("\n=== 各機組負荷率分析 ===")
            for unit_name, unit_result in load_analysis.items():
                self.result_text.append(f"\n◆ {unit_name}")
                stats = safe_get_stats(unit_result)
                self.result_text.append(f"  平均負荷率: {safe_fmt(stats.get('mean'), '.1f')}%")
                self.result_text.append(f"  最大負荷率: {safe_fmt(stats.get('max'), '.1f')}%")
                self.result_text.append(f"  最小負荷率: {safe_fmt(stats.get('min'), '.1f')}%")
                self.result_text.append(
                    f"  低負荷率時間割合(<30%): {safe_fmt(stats.get('low_load_percentage'), '.1f')}%")
                self.result_text.append(
                    f"  高負荷率時間割合(>80%): {safe_fmt(stats.get('high_load_percentage'), '.1f')}%")
                # 新增：外气温相关性
                if 'weather_corr' in unit_result and unit_result['weather_corr'] is not None:
                    self.result_text.append(f"  外気温度との相関性: {safe_fmt(unit_result.get('weather_corr'), '.2f')}")
                # 新增：AI分析
                analysis_text = str(unit_result.get('analysis', '') or '').strip()
                if analysis_text and "分析不可用" not in analysis_text:
                    self.result_text.append("  ◆ 専門分析:")
                    self.result_text.append(analysis_text)
                elif analysis_text:
                    self.result_text.append(f"  {analysis_text}")
        elif self.current_results.get('config_used', {}).get('analyze_load', False):
            self.result_text.append("\n=== 負荷率分析 ===")
            self.result_text.append("負荷率分析が利用できません（電力データ列または冷却能力値が不足しています）")

        # 显示气象数据状态 (新增)
        if self.current_results.get('weather_data'):
            self.result_text.append("\n=== 気象データ ===")
            self.result_text.append("外気温度データを正常に取得し、分析に統合しました")
        # 显示COP分析结果 (新增)
        cop_analysis_multi = self.current_results.get('cop_analysis')
        if cop_analysis_multi and isinstance(cop_analysis_multi, dict):
            self.result_text.append("\n=== 各機組COP分析 ===")
            for unit_name, unit_result in cop_analysis_multi.items():
                heat_source_cop = unit_result.get('heat_source_cop', {})
                stats = (heat_source_cop or {}).get('cop_stats', {})
                self.result_text.append(f"\n◆ {unit_name}")
                self.result_text.append(f"  平均COP: {safe_fmt(stats.get('cop_mean'), '.2f')}")
                self.result_text.append(f"  最大COP: {safe_fmt(stats.get('cop_max'), '.2f')}")
                self.result_text.append(f"  最小COP: {safe_fmt(stats.get('cop_min'), '.2f')}")
                self.result_text.append(f"  標準偏差: {safe_fmt(stats.get('cop_std'), '.2f')}")
                self.result_text.append(
                    f"  高効率時間割合(COP>5): {safe_fmt(stats.get('high_cop_percentage'), '.1f')}%")
                self.result_text.append(f"  低効率時間割合(COP<2): {safe_fmt(stats.get('low_cop_percentage'), '.1f')}%")
                # 显示AI报告
                cop_report = (heat_source_cop or {}).get('cop_report', '')
                if cop_report:
                    self.result_text.append("  ◆ 専門AIレポート：\n" + cop_report)
                # 显示用到的列
                used_cols = (heat_source_cop or {}).get('columns_used', {})
                if used_cols:
                    self.result_text.append("  使用列:\n" + pprint.pformat(used_cols, indent=2, width=60))
        else:
            self.result_text.append("\n=== COP分析 ===")
            self.result_text.append("COP分析が利用できません（温度、流量または電力データ列が不足しています）")

        # 显示分组信息
        self.result_text.append("=== グループ詳細 ===")
        groupings = self.current_results.get('groupings', {})
        for group_name, group_info in groupings.items():
            self.result_text.append(f"\n◆ {group_name}")
            self.result_text.append(f"  説明: {group_info.get('description', '説明無し')}")
            self.result_text.append(f"  含まれる列: {', '.join(group_info.get('columns', []))}")

        # 显示图表信息
        if charts := self.current_results.get('charts', []):
            self.result_text.append("\n=== 生成グラフ ===")
            for chart in charts:
                self.result_text.append(f"· {chart}")

        # 显示运行分析结果
        if operation_analysis := self.current_results.get('operation_analysis', {}):
            self.result_text.append("\n=== 運転状態分析 ===")
            for unit, analysis in operation_analysis.items():
                self.result_text.append(f"\n◆ {unit}の分析:")
                self.result_text.append(analysis)

        # 滚动到顶部
        self.result_text.moveCursor(QtGui.QTextCursor.Start)

    def run_analysis(self):
        file_path = self.file_path_edit.text()
        if not file_path:
            QMessageBox.warning(self, "警告", "データファイルを選択してください！")
            return

        # === 新增验证1：检查气象数据配置 ===
        if self.enable_weather_check.isChecked():
            try:
                lat = float(self.lat_edit.text())
                lng = float(self.lng_edit.text())
                if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                    raise ValueError("纬度必须在-90~90之间，经度必须在-180~180之间")
            except ValueError as e:
                QMessageBox.warning(self, "入力エラー", f"無効な緯度・経度:\n{str(e)}")
                return

        # === 新增验证2：检查负荷分析配置 ===
        if self.enable_load_check.isChecked() and not self.cooling_capacity_edit.text():
            QMessageBox.warning(self, "入力エラー", "負荷率分析を有効にする場合は冷却能力(kW)を入力してください")
            return

        try:
            # 设置matplotlib日文字体
            import matplotlib.pyplot as plt
            plt.rcParams['font.sans-serif'] = ['MS Gothic']
            plt.rcParams['axes.unicode_minus'] = False

            config = {
                'use_ai': self.use_ai_check.isChecked(),
                'group_mode': "ai",
                'custom_rule': self.custom_rule_edit.text() if self.group_mode.currentIndex() == 3 else None,
                'unit_pattern': self.unit_pattern_edit.text(),
                'chart_type': self.chart_type.currentIndex(),
                'lang': 'JP',
                'output_dir': 'output',
                'generate_charts': True,
                # 新增配置
                'analyze_load': self.enable_load_check.isChecked(),
                'cooling_capacity': float(self.cooling_capacity_edit.text())
                if self.enable_load_check.isChecked() and self.cooling_capacity_edit.text()
                else None,
                'fetch_weather': self.enable_weather_check.isChecked(),
                'location': (
                    float(self.lat_edit.text()) if self.lat_edit.text() else None,
                    float(self.lng_edit.text()) if self.lng_edit.text() else None
                ),
                'analyze_cop': self.enable_cop_check.isChecked(),
                'water_specific_heat': float(self.water_shc_edit.text()) if self.water_shc_edit.text() else 4.18,
                'analyze_columns': self.analyze_columns_check.isChecked(),
            }

            # === 读文件 ===
            df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
            import re
            def _canon_col(s: str) -> str:
                s = str(s)
                s = s.replace('\u3000', ' ')  # 全角空格 → 半角
                s = s.replace('\ufeff', '')  # BOM
                s = re.sub(r'\s+', ' ', s)  # 连续空格合并
                return s.strip()

            raw2canon = {c: _canon_col(c) for c in df.columns}
            canon2raw = {v: k for k, v in raw2canon.items()}
            df.rename(columns=raw2canon, inplace=True)

            # 存在 config 里给后续用
            config['col_name_maps'] = {'raw2canon': raw2canon, 'canon2raw': canon2raw}

            # === 找时间列 ===
            time_keywords = ['time', 'date', 'datetime', '時', '日', '日時', '时间', '日期']

            def _is_time_col(c):
                s = str(c).lower()
                return any(k in s for k in time_keywords)

            time_cols = [c for c in df.columns if _is_time_col(c)]
            if not time_cols:
                raise ValueError("数据文件中未找到有效的时间列（需包含'time'/'日時'/'日期' 等关键字的列名）")
            time_col = time_cols[0]

            # 统一时间类型
            def parse_time_series(s):
                import pandas as pd
                if pd.api.types.is_numeric_dtype(s):
                    out = pd.to_datetime(s, unit='d', origin='1899-12-30', errors='coerce')
                    if out.notna().any():
                        return out
                fmts = [
                    '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
                    '%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M',
                    '%Y-%m-%d', '%Y/%m/%d',
                    '%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M', '%m/%d/%Y'
                ]
                s = s.astype(str)
                for fmt in fmts:
                    out = pd.to_datetime(s, format=fmt, errors='coerce')
                    if out.notna().mean() > 0.8:
                        return out
                return pd.to_datetime(s, errors='coerce')

            df[time_col] = parse_time_series(df[time_col])  # ← 用这个替换原来的 to_datetime




            # === 如需做负荷分析，先做列角色识别（基于“已补全”的 df）===
            if config.get('analyze_load', False):
                try:
                    _client = getattr(self, "openai_client", None)  # 允许是 None

                    # 从 GUI 复选框读取是否导出C1–C8
                    export_slots = self.export_slots_check.isChecked()

                    # 可选：生成一个带时间戳的文件名，避免每次覆盖
                    slot_csv_path = None
                    if export_slots:
                        # 你指定的绝对路径
                        base_dir = r"D:\dynamoFile\PythonScript\LLM\output\slot"
                        os.makedirs(base_dir, exist_ok=True)

                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        data_name = os.path.splitext(os.path.basename(file_path))[0]
                        slot_csv_path = os.path.join(base_dir, f"slot_scores_{data_name}_{ts}.csv")

                    from analysis_core import slot_unit_infer

                    review_dict, ai_roles,slot_details = batch_physical_role_review(
                        df,
                        client=_client,
                        export_slot_csv=export_slots,
                        slot_csv_path=slot_csv_path,
                        unit_infer=slot_unit_infer,  # ★
                    )
                    config['ai_roles'] = ai_roles
                    config['slot_details'] = slot_details
                    config['review_dict'] = review_dict

                except Exception as e:
                    print("[LLM列识别失败] 回退到关键字尝试：", e)
                    ai_roles = {
                        col: ("heat_source_power" if ("電力(kW)" in str(col) or " power(kw)" in str(col).lower())
                              else "other") for col in df.columns
                    }

                # 经过 ensure_power_columns 之后，可能已经有了電力(kW)
                has_power = any(role in ("heat_source_power", "fan_power") for role in ai_roles.values()) \
                            or any(("電力(kW)" in str(c)) or (" power(kw)" in str(c).lower()) for c in df.columns)
                has_energy = any(("電力量" in str(c)) or ("energy" in str(c).lower()) for c in df.columns)

                if not has_power and not has_energy:
                    QMessageBox.warning(self, "警告",
                                        "電力/電力量データ列が見つかりません（LLM分類でも未検出）。"
                                        "列名に『電力(kW)』または『電力量(kWh)』相当の項目が必要です。")
                    return

                config['ai_roles'] = ai_roles

            # === 把预处理好的 df 和 time_col 传给核心 ===
            config['time_col'] = time_col
            config['preloaded_df'] = df

            weather_df = None
            if config['fetch_weather']:
                try:
                    from analysis_core import fetch_weather_data
                    lat, lng = config['location']

                    # 直接用上面已经转成 datetime 的 df[time_col]
                    start_date = pd.to_datetime(df[time_col]).min().strftime("%Y-%m-%d")
                    end_date = pd.to_datetime(df[time_col]).max().strftime("%Y-%m-%d")

                    weather_df = fetch_weather_data((lat, lng), start_date, end_date)
                    self.current_results = getattr(self, "current_results", {})
                    self.current_results['weather_data'] = weather_df
                    config['weather_df'] = weather_df
                except Exception as e:
                    print("气象数据获取失败：", e)
                    config['weather_df'] = None
            else:
                config['weather_df'] = None

            # 调用分析函数
            self.current_results = analyze_data(file_path, config)
            # print("DEBUG: analyze_data 返回 self.current_results =", self.current_results)
            # print("DEBUG: self.current_results.get('cop_analysis') =", self.current_results.get('cop_analysis'))
            # 显示结果
            self.display_results()
            self.statusBar().showMessage("分析完了！")


        except ValueError as e:  # 专门捕获输入错误
            QMessageBox.warning(self, "入力エラー", str(e))
        except Exception as e:  # 其他错误
            QMessageBox.critical(self, "エラー", f"分析中にエラーが発生しました:\n{str(e)}")
            print(f"DEBUG: 完整错误信息:\n{traceback.format_exc()}")  # 打印完整错误日志
            self.statusBar().showMessage(f"分析失敗: {str(e)}")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HVACAnalyzerGUI()
    window.show()
    sys.exit(app.exec_())