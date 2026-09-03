"""Translate evidence-backed engineering findings into actionable user guidance."""
from __future__ import annotations

from dataclasses import dataclass

from building_ai.models import DiagnosticFinding, EnergySavingOpportunity


@dataclass(frozen=True, slots=True)
class UserInterpretation:
    finding_id: str
    equipment_id: str | None
    problem: str
    explanation: str
    actions: tuple[str, ...]
    priority: str
    implementation_difficulty: str
    cost_level: str
    expected_effect: str
    evidence_strength: str
    risk_level: int
    technical_details: str


class InterpretationService:
    """No recommendation is produced without a formal Finding.

    Wording reflects evidence strength: Level 1 is inspection, Level 2 is a
    small, observed site adjustment, and Level 3 requires an engineering and
    economic assessment.  The service does not claim savings percentages.
    """

    def interpret(self, finding: DiagnosticFinding, opportunity: EnergySavingOpportunity | None = None, *, language: str = "zh_CN") -> UserInterpretation:
        chinese = language == "zh_CN"
        evidence = finding.evidence[0].metric_value if finding.evidence else {}
        detail = self._technical(finding, evidence, chinese)
        if finding.finding_type == "low_chilled_water_delta_t":
            return UserInterpretation(
                finding.finding_id, finding.equipment_id,
                "冷冻水利用效率偏低" if chinese else "Chilled-water utilization is low",
                "设备运行时供水和回水的温度差经常偏小。这通常表示水循环量与实际冷量需求可能不匹配，也可能与旁通、阀门或末端换热有关。" if chinese else "The supply-to-return water temperature difference is often small during operation. Water flow may not match demand, or bypass, valve, or terminal heat-transfer conditions may contribute.",
                (("先检查冷冻水泵频率是否长期偏高。" if chinese else "First check whether chilled-water pump speed is persistently high."),
                 ("检查旁通阀是否过度开启，以及末端阀门是否正常动作。" if chinese else "Check bypass opening and terminal-valve operation."),
                 ("在室内舒适度正常的前提下，只能小幅降低流量，并同时观察供回水温差、室温和主机功率。" if chinese else "Only after confirming comfort, make a small flow reduction and observe ΔT, room temperature, and plant power."),
                 ("检查过滤器、盘管和末端换热设备是否堵塞或换热不足。" if chinese else "Inspect filters, coils, and terminal heat-transfer equipment for fouling or insufficient heat transfer.")),
                "P1", "低～中" if chinese else "Low–medium", "低～中" if chinese else "Low–medium",
                "有助于减少水泵能耗并改善主机运行效率；需要现场验证。" if chinese else "May reduce pump energy and improve plant efficiency; site verification is required.",
                "高" if finding.confidence >= .8 and chinese else "中" if chinese else ("High" if finding.confidence >= .8 else "Medium"), 2, detail,
            )
        templates_zh = {
            "low_heat_source_cop": ("设备运行效率偏低", "设备在有效运行时消耗的电力相对较多。", ("检查换热器清洁度、冷却侧条件和设定值。", "先核对低负荷时的运行台数和启停顺序。", "任何设定值调整都应小幅进行并观察舒适度和功率。"), "提高设备效率并减少不必要耗电。", "P1", 2),
            "off_hour_operation": ("非工作时段仍在运行", "设备在已配置的工作时间之外仍检测到运行。", ("先核对运行时间表、节假日和手动覆盖。", "确认无必要负荷后，再逐步优化停机顺序。"), "减少不必要运行时间和能耗。", "P1", 1),
            "low_load_high_power": ("低负荷时耗电偏高", "设备负荷较低时仍保持较高耗电，可能存在运行组合或控制策略不匹配。", ("检查当前运行台数和最低负荷限制。", "现场确认后，优化设备分级、变频和启停顺序。"), "减少低负荷低效率运行。", "P2", 2),
            "frequent_start_stop": ("设备启停过于频繁", "设备反复启动和停止，可能增加磨损并降低运行稳定性。", ("检查最小运行/停机时间和控制死区。", "检查传感器稳定性及设备分级逻辑。"), "减少设备磨损并提升稳定性。", "P2", 2),
            "parallel_heat_source_operation": ("多台设备同时运行需要复核", "多台设备同时运行本身不一定异常，但应结合实际负荷确认群控策略。", ("核对总负荷、备用需求及设备容量。", "现场确认后再调整群控或启停顺序。"), "避免不必要的并联运行。", "P2", 2),
        }
        templates_en = {
            "low_heat_source_cop": ("Equipment efficiency is low", "The equipment uses relatively high input power during valid operation.", ("Inspect heat-exchanger cleanliness, heat-rejection conditions, and setpoints.", "Check the number and staging order of units running at low load.", "Make only small setpoint changes and observe comfort and power."), "May improve equipment efficiency and reduce avoidable power use; site verification is required.", "P1", 2),
            "off_hour_operation": ("Operation outside the configured schedule", "Positive equipment power was measured outside the configured operating hours.", ("Verify schedules, holidays, and manual overrides first.", "After confirming there is no required load, review the shutdown sequence."), "May reduce avoidable runtime and energy use; site verification is required.", "P1", 1),
            "low_load_high_power": ("Power use is high at low load", "Measured power remains high while calculated load is low; staging or control may not match demand.", ("Check the number of operating units and minimum-load limits.", "After site verification, review staging, variable-speed control, and start/stop sequencing."), "May reduce inefficient low-load operation; site verification is required.", "P2", 2),
            "frequent_start_stop": ("Equipment starts and stops frequently", "Repeated start/stop events may increase wear and reduce operating stability.", ("Check minimum run/off times and control deadbands.", "Check sensor stability and staging logic."), "May reduce equipment wear and improve stability; site verification is required.", "P2", 2),
            "parallel_heat_source_operation": ("Parallel equipment operation needs review", "Parallel operation is not inherently a fault; measured load and staging intent should be checked.", ("Verify total load, redundancy requirements, and equipment capacity.", "Only after site confirmation, review staging and start/stop sequencing."), "May avoid unnecessary parallel operation; site verification is required.", "P2", 2),
        }
        templates = templates_zh if chinese else templates_en
        fallback = (("运行情况需要关注", "检测到需要现场核实的运行异常。", ("核对相关点位、设备状态和现场运行条件。",), "需要现场验证。", "P2", 1) if chinese else ("Operation needs attention", "A measured operating condition requires site review.", ("Verify the related signals, equipment state, and site operating conditions.",), "Requires site verification.", "P2", 1))
        problem, explanation, actions, effect, priority, risk = templates.get(finding.finding_type, fallback)
        return UserInterpretation(finding.finding_id, finding.equipment_id, problem, explanation, tuple(actions), priority, "中" if chinese else "Medium", "低～中" if chinese else "Low–medium", effect + (" 需要现场验证。" if chinese and "验证" not in effect else ""), "高" if chinese and finding.confidence >= .8 else "中" if chinese else ("High" if finding.confidence >= .8 else "Medium"), risk, detail)

    @staticmethod
    def _technical(finding: DiagnosticFinding, evidence: dict, chinese: bool) -> str:
        period = finding.affected_period or {}
        if chinese:
            return f"诊断代码：{finding.finding_type}\n有效样本：{finding.valid_sample_count}\n发生次数：{finding.occurrence_count}\n置信度：{finding.confidence:.0%}\n分析时段：{period.get('start') or '—'} ～ {period.get('end') or '—'}\n证据：{evidence}"
        return f"Finding code: {finding.finding_type}\nValid samples: {finding.valid_sample_count}\nOccurrences: {finding.occurrence_count}\nConfidence: {finding.confidence:.0%}\nPeriod: {period.get('start') or '—'} to {period.get('end') or '—'}\nEvidence: {evidence}"
