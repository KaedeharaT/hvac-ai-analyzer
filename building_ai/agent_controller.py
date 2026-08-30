"""Read-only, tool-grounded conversation controller for BuildingAI."""
from __future__ import annotations

import json
import re
from typing import Callable

from building_ai.i18n import LanguageManager, tr
from building_ai.llm.prompts import AGENT_SYSTEM_PROMPT
from building_ai.core.equipment_identity import normalize_equipment_id


class AgentController:
    def __init__(self, context):
        self.context = context

    def answer(self, message: str, tool_event: Callable[[str, str], None] | None = None, llm_event: Callable[[dict], None] | None = None) -> str:
        project = self.context.current_project
        text = message.casefold()
        chinese = LanguageManager.instance().language == "zh_CN"
        identity_question = any(token in text for token in ("你是谁", "你能做什么", "who are you", "what can you do"))
        provider = self.context.llm_manager.get_provider()
        if identity_question:
            if provider.is_configured:
                try:
                    response = self._generate(provider,
                        "Briefly introduce yourself as BuildingAI's read-only BEMS and HVAC analytics assistant. Do not mention project data status.",
                        "Answer in Simplified Chinese only." if chinese else "Answer in English only.", llm_event,
                    ).strip()
                    response = re.sub(r"^\s*ABSTAIN\s*[:：-]?\s*", "", response, flags=re.IGNORECASE)
                    if response:
                        return response
                except Exception:
                    pass
            return "我是 BuildingAI 的 AI 助手，可以基于当前项目的真实 BEMS 数据提供只读分析与解释。" if chinese else "I am the BuildingAI assistant. I provide read-only analysis and explanations from the current project's real BEMS data."
        equipment = self._equipment_name(message)
        tool_names: list[str] = []
        project_question = any(token in text for token in ("总结", "summary", "运行情况", "当前项目", "项目有什么数据", "当前有什么数据", "当前数据", "project data", "项目数据"))
        equipment_list_question = any(token in text for token in ("哪些设备", "设备有哪些", "what equipment", "equipment list"))
        energy_question = any(token in text for token in ("能耗", "能源", "energy", "power", "功率", "负荷", "昨天", "最近", "peak", "峰值"))
        temperature_data_question = any(token in text for token in ("温度趋势", "temperature trend", "室外温度", "outdoor temperature"))
        lowest_cop_question = (("cop" in text or "能效比" in text) and any(token in text for token in ("最低", "最差", "lowest", "worst")))
        opportunity_question = any(token in text for token in ("节能机会", "节能方案", "节能建议", "energy saving", "saving opportunity", "opportunit"))
        finding_question = any(token in text for token in ("异常", "问题", "诊断", "finding", "fault"))
        data_question = any((project_question, equipment_list_question, energy_question, temperature_data_question, lowest_cop_question, opportunity_question, finding_question, equipment is not None, "语义" in text, "semantic" in text))
        # General engineering questions deliberately stay outside the project
        # evidence path.  A question about ΔT/COP is not automatically a claim
        # about the selected building.
        general_engineering = not data_question and any(token in text for token in (
            "一般", "常见", "原理", "作用", "方法", "怎么", "why", "how", "common", "cause", "causes",
            "冷冻水", "delta t", "δt", "cop", "变流量", "变频", "vfd", "hvac", "暖通", "节能",
        ))
        if not data_question:
            return self._general_answer(message, chinese, provider, general_engineering, llm_event)
        if project is None:
            return "请先打开项目后再查询项目数据。" if chinese else "Please open a project before querying project data."
        if project_question:
            tool_names += ["get_project_summary", "get_equipment_summary", "get_analysis_results", "get_energy_summary"]
        if equipment_list_question:
            tool_names += ["get_project_summary", "get_equipment_summary"]
        if energy_question:
            tool_names += ["get_energy_summary", "get_energy_timeseries", "get_temperature_summary"]
        if temperature_data_question:
            tool_names += ["get_temperature_summary", "get_energy_timeseries"]
        if lowest_cop_question:
            tool_names += ["get_analysis_results", "get_equipment_kpis"]
        if equipment:
            tool_names += ["get_equipment_summary", "get_equipment_kpis", "get_diagnostic_findings"]
        if finding_question:
            tool_names += ["get_diagnostic_findings", "get_analysis_results"]
        if opportunity_question:
            tool_names += ["get_diagnostic_findings", "get_energy_opportunities", "get_analysis_results", "get_energy_summary"]
        if any(token in text for token in ("语义", "semantic")):
            tool_names.append("get_semantic_mapping")
        evidence = {}
        for name in dict.fromkeys(tool_names):
            if tool_event:
                tool_event(name, "RUNNING")
            kwargs = {"project_id": project.project_id}
            if name in {"get_equipment_summary", "get_diagnostic_findings"} and equipment:
                kwargs["equipment_name"] = equipment
            if name == "get_energy_timeseries":
                kwargs["range_key"] = "24h" if "24" in text else "7d" if any(token in text for token in ("最近", "一周", "7天", "7d", "week")) else "30d" if any(token in text for token in ("30天", "30d", "month")) else "all"
            result = self.context.agent.tools.call(name, **kwargs)
            evidence[name] = result.data if result.ok else {"error": result.error}
            if tool_event:
                tool_event(name, "SUCCESS" if result.ok else "FAILED")
        if tool_names:
            # Data questions are deterministically rendered from formal tool
            # evidence.  An LLM is never allowed to override a non-empty result.
            return self._grounded_data_answer(message, equipment, chinese, evidence)
        return self._general_answer(message, chinese, provider, general_engineering, llm_event)

    def _grounded_data_answer(self, message: str, equipment: str | None, chinese: bool, evidence: dict) -> str:
        text = message.casefold()
        summary = evidence.get("get_project_summary", {})
        if isinstance(summary, dict) and summary.get("data_available") is False:
            return "当前项目没有可读取的持久化数据。" if chinese else "The current project has no readable persisted data."
        if equipment:
            return self._grounded_fallback(equipment, chinese)
        if (("cop" in text or "能效比" in text) and any(token in text for token in ("最低", "最差", "lowest", "worst"))):
            return self._lowest_cop_answer(chinese)
        if any(token in text for token in ("节能机会", "节能方案", "节能建议", "energy saving", "saving opportunity", "opportunit")):
            opportunities = evidence.get("get_energy_opportunities", {})
            rows = opportunities.get("opportunities", []) if isinstance(opportunities, dict) else []
            if rows:
                if chinese:
                    return "根据当前项目已验证的 Finding，建议：\n" + "\n".join(
                        f"- {item.get('equipment_name') or '项目'}：{self._opportunity_text(item, 'title')}。{self._opportunity_text(item, 'description')}"
                        for item in rows
                    )
                return "Based on the current project's validated findings:\n" + "\n".join(
                    f"- {item.get('equipment_name') or 'Project'}: {item.get('title')}. {item.get('recommendation')}"
                    for item in rows
                )
            return self._general_engineering_fallback(chinese, project_data_unavailable=True)
        energy = evidence.get("get_energy_summary", {})
        equipment_rows = evidence.get("get_equipment_summary", [])
        if any(token in text for token in ("最近", "一周", "7天", "7d", "趋势", "trend")):
            trend = evidence.get("get_energy_timeseries", {})
            payload = trend.get("result", {}) if isinstance(trend, dict) else {}
            points = next(iter(payload.get("series", [])), {}).get("data", [])
            values = [item.get("value") for item in points if isinstance(item.get("value"), (int, float))]
            if len(values) >= 2:
                half = max(1, len(values) // 2); previous = sum(values[:half]) / half; current = sum(values[half:]) / max(1, len(values[half:]))
                direction = ("上升" if current > previous else "下降" if current < previous else "基本持平") if chinese else ("increased" if current > previous else "decreased" if current < previous else "remained broadly stable")
                label = payload.get("requested_range", "available range")
                return (f"根据当前项目正式能源时序，{label} 内后半段平均能耗为 {current:.2f} kWh/周期，前半段为 {previous:.2f} kWh/周期，趋势{direction}。"
                        if chinese else f"From the formal current-project energy time series, the latter half of {label} averages {current:.2f} kWh per period versus {previous:.2f}; the trend {direction}.")
            return "当前项目没有足够的有效能源时序来判断该范围的趋势。" if chinese else "The current project has insufficient valid energy time series to determine that trend."
        if chinese:
            lines = []
            if isinstance(summary, dict):
                lines.append(f"项目 {summary.get('project_name')}：已读取持久化数据，共 {summary.get('number_of_rows', 0)} 行、{summary.get('number_of_points', 0)} 个点位。")
                names = summary.get("discovered_equipment", []); lines.append("设备：" + ("、".join(names) if names else "当前未发现可确认设备。"))
                lines.append(f"时间范围：{summary.get('time_range', {}).get('start') or '—'} ～ {summary.get('time_range', {}).get('end') or '—'}；采样间隔：{summary.get('sampling_interval') or '—'}。")
            if isinstance(energy, dict) and energy.get("summary"):
                values = energy["summary"]
                def number(key):
                    value = values.get(key)
                    return f"{value:.2f}" if isinstance(value, (int, float)) else "—"
                lines.append(f"能源：总能耗 {number('total_energy_kwh')} kWh，峰值功率 {number('peak_power_kw')} kW，平均 COP {number('average_cop')}。")
            return "\n".join(lines) or "当前项目没有可用的分析结果。"
        lines = []
        if isinstance(summary, dict):
            lines.append(f"Project {summary.get('project_name')}: persisted data is available with {summary.get('number_of_rows', 0)} rows and {summary.get('number_of_points', 0)} points.")
            lines.append("Equipment: " + (", ".join(summary.get("discovered_equipment", [])) or "no confirmed equipment."))
        if isinstance(energy, dict) and energy.get("summary"):
            values = energy["summary"]; lines.append(f"Energy: total {values.get('total_energy_kwh') if values.get('total_energy_kwh') is not None else '—'} kWh; peak {values.get('peak_power_kw') if values.get('peak_power_kw') is not None else '—'} kW; average COP {values.get('average_cop') if values.get('average_cop') is not None else '—'}.")
        return "\n".join(lines) or "No current project analysis is available."

    @staticmethod
    def _opportunity_text(row: dict, part: str) -> str:
        key = f"opportunity_{str(row.get('category', '')).lower()}_{'title' if part == 'title' else 'description'}"
        translated = tr(key)
        if translated != key:
            return translated
        return str(row.get('title' if part == 'title' else 'recommendation', ""))

    def _lowest_cop_answer(self, chinese: bool) -> str:
        diagnosis = self.context.diagnosis_result
        kpis = diagnosis.analytics.equipment_kpis if diagnosis and diagnosis.analytics else []
        candidates = []
        for item in kpis:
            value = item.metric_summary.get("cop", {}).get("mean")
            if item.status == "available" and isinstance(value, (int, float)):
                candidates.append((float(value), item))
        if not candidates:
            return "当前项目缺少可可靠计算 COP 的设备数据。" if chinese else "The current project has no equipment with a reliably calculated COP."
        value, item = min(candidates, key=lambda row: row[0])
        return (f"根据当前项目的正式 KPI，COP 最低的是 {item.equipment_name}，平均 COP 为 {value:.2f}。"
                if chinese else f"According to the formal current-project KPIs, {item.equipment_name} has the lowest average COP: {value:.2f}.")

    def _general_answer(self, message: str, chinese: bool, provider, engineering: bool, llm_event: Callable[[dict], None] | None = None) -> str:
        """Answer chat and engineering-knowledge questions without tool gating."""
        if provider.is_configured:
            language = "Simplified Chinese only" if chinese else "English only"
            scope = (
                "This is general HVAC/energy engineering knowledge, not a diagnosis of the current project. State that distinction when giving recommendations."
                if engineering else
                "This is normal BuildingAI product conversation. Do not discuss current project data unless the user asks for it."
            )
            prompt = f"Answer in {language}. {scope}\nUser question: {message}"
            try:
                response = self._generate(provider, prompt, "You are BuildingAI's helpful, read-only BEMS and HVAC analytics assistant. Do not output ABSTAIN.", llm_event).strip()
                response = re.sub(r"^\s*ABSTAIN\s*[:：-]?\s*", "", response, flags=re.IGNORECASE)
                if response and (not chinese or re.search(r"[\u4e00-\u9fff]", response)):
                    return response
            except Exception:
                pass
        if engineering:
            return self._general_engineering_fallback(chinese)
        return ("我是 BuildingAI 的 AI 助手。我可以解释软件能力、提供通用 HVAC/能源工程知识，并基于当前项目的真实持久化数据回答设备、能耗、KPI、诊断和节能机会问题。"
                if chinese else "I am the BuildingAI assistant. I can explain the product, provide general HVAC and energy-engineering guidance, and answer equipment, energy, KPI, diagnosis, and opportunity questions from the selected project's persisted data.")

    @staticmethod
    def _generate(provider, prompt: str, system_prompt: str, llm_event: Callable[[dict], None] | None) -> str:
        provider_name=getattr(provider, 'display_name', type(provider).__name__)
        try:
            response=provider.generate(prompt, system_prompt=system_prompt, temperature=0)
        except Exception as exc:
            if llm_event: llm_event({'provider':provider_name,'operation':'generate','status':'FAILED','error_type':type(exc).__name__})
            raise
        if llm_event: llm_event({'provider':provider_name,'operation':'generate','status':'SUCCEEDED'})
        return response

    @staticmethod
    def _general_engineering_fallback(chinese: bool, project_data_unavailable: bool = False) -> str:
        if chinese:
            prefix = "当前项目没有足够的已验证证据形成项目级节能方案；以下为通用 HVAC 工程建议，并非当前项目诊断结论。\n" if project_data_unavailable else "以下为通用 HVAC 工程知识，不是当前项目的诊断结论。\n"
            return prefix + "冷冻水 ΔT 偏低常见于流量偏大、旁通或三通阀泄漏、末端换热不足、阀门控制不稳定，以及设定值或压差控制不当。应先核对供回水温度、流量、阀位、旁通和末端负荷；再在现场确认后优化泵变频、压差设定、阀门和机组群控。COP 偏低还应检查冷凝/蒸发换热、污垢、冷却侧条件、部分负荷和机组启停策略。"
        prefix = "The current project has insufficient validated evidence for a project-specific saving plan; the following is general HVAC engineering guidance, not a project diagnosis.\n" if project_data_unavailable else "The following is general HVAC engineering guidance, not a diagnosis of the current project.\n"
        return prefix + "Low chilled-water ΔT commonly relates to excessive flow, bypass or leaking three-way valves, insufficient terminal heat transfer, unstable valve control, or unsuitable setpoints/differential-pressure control. Verify supply/return temperatures, flow, valve position, bypasses, and terminal load before optimizing pump VFD control, pressure setpoints, valves, or staging. Low COP also warrants checking condenser/evaporator heat exchange, fouling, cooling-side conditions, part-load operation, and staging."

    def _safe_response(self, response: str, chinese: bool, evidence: dict) -> bool:
        if not response:
            return False
        if chinese and not re.search(r"[\u4e00-\u9fff]", response):
            return False
        # Do not permit a model to invent a low-COP diagnosis when the tool
        # evidence contains no such finding (a previously observed failure).
        source = json.dumps(evidence, ensure_ascii=False).casefold()
        lowered = response.casefold()
        if "low_heat_source_cop" not in source and any(token in lowered for token in ("low heat source cop", "cop is low", "cop偏低", "cop 为低")):
            return False
        return True

    def _grounded_fallback(self, equipment: str | None, chinese: bool) -> str:
        diagnosis = self.context.diagnosis_result
        if diagnosis is None or diagnosis.analytics is None:
            return "当前项目尚未运行分析。" if chinese else "Analysis has not been run for the current project."
        kpis = diagnosis.analytics.equipment_kpis
        names = {item.equipment_id: item.equipment_name for item in kpis}
        if equipment:
            kpi = next((item for item in kpis if item.equipment_name.casefold() == equipment.casefold()), None)
            if kpi is None:
                return f"未找到设备 {equipment} 的分析结果。" if chinese else f"No analysis result was found for {equipment}."
            metrics = kpi.metric_summary
            values = lambda key: metrics.get(key, {}).get("mean")
            if chinese:
                lines = [f"根据当前项目的已验证分析结果，{kpi.equipment_name}：", f"- 状态：{'可用' if kpi.status == 'available' else '无法计算'}", f"- 有效样本：{kpi.valid_sample_count}"]
                if kpi.status == "available":
                    lines += [f"- 平均 COP：{values('cop'):.2f}", f"- 平均 ΔT：{values('delta_t_c'):.2f}°C", f"- 平均流量：{values('flow_lps'):.2f} L/s", f"- 平均输入功率：{values('power_kw'):.2f} kW", f"- 平均热负荷：{values('thermal_load_kw'):.2f} kW"]
                related = [item for item in diagnosis.findings if item.equipment_id == kpi.equipment_id]
                if related:
                    finding = related[0]; user = self.context.interpretation.interpret(finding, language="zh_CN")
                    evidence = finding.evidence[0].metric_value if finding.evidence else {}
                    return "\n".join([
                        f"{kpi.equipment_name}：{user.problem}", user.explanation,
                        "建议先做：" + user.actions[0],
                        "建议操作：" + "；".join(user.actions[1:]),
                        f"优先级：{user.priority}。预期作用：{user.expected_effect}",
                        f"技术数据：平均 COP {values('cop'):.2f}；平均 ΔT {values('delta_t_c'):.2f}°C；有效样本 {kpi.valid_sample_count}；证据 {evidence}。",
                    ])
                else: lines.append("- 诊断：当前未触发确定性的诊断结果。")
                return "\n".join(lines)
            lines = [f"Validated result for {kpi.equipment_name}:", f"- Status: {'available' if kpi.status == 'available' else 'unavailable'}", f"- Valid samples: {kpi.valid_sample_count}"]
            if kpi.status == "available": lines += [f"- Average COP: {values('cop'):.2f}", f"- Average ΔT: {values('delta_t_c'):.2f} °C", f"- Average flow: {values('flow_lps'):.2f} L/s", f"- Average input power: {values('power_kw'):.2f} kW", f"- Average thermal load: {values('thermal_load_kw'):.2f} kW"]
            related = [item for item in diagnosis.findings if item.equipment_id == kpi.equipment_id]
            if related:
                finding = related[0]; user = self.context.interpretation.interpret(finding, language="en_US")
                return "\n".join([f"{kpi.equipment_name}: {user.problem}", user.explanation, "Start with: " + user.actions[0], "Recommended actions: " + "; ".join(user.actions[1:]), f"Priority: {user.priority}. Expected effect: {user.expected_effect}", f"Technical data: average COP {values('cop'):.2f}; average ΔT {values('delta_t_c'):.2f} °C; valid samples {kpi.valid_sample_count}."])
            lines.append("- Finding: " + (", ".join(item.finding_type for item in related) if related else "no deterministic finding triggered."))
            return "\n".join(lines)
        findings = diagnosis.findings
        validation = diagnosis.consistency_validation.passed if diagnosis.consistency_validation else False
        if chinese:
            lines = [f"根据当前项目的已验证分析结果：一致性校验{'通过' if validation else '未通过'}。", f"共分析 {len(kpis)} 台设备；诊断发现 {len(findings)} 项；节能机会 {len(self.context.opportunities)} 项。"]
            for item in kpis:
                metric = item.metric_summary
                lines.append(f"- {item.equipment_name}：COP {metric.get('cop', {}).get('mean', 0):.2f}，ΔT {metric.get('delta_t_c', {}).get('mean', 0):.2f}°C，状态：{'可用' if item.status == 'available' else '无法计算'}。")
            for finding in findings:
                evidence = finding.evidence[0].metric_value if finding.evidence else {}
                lines.append(f"- {names.get(finding.equipment_id, finding.equipment_id)}：{tr(f'finding_{finding.finding_type}_title')}，低 ΔT 占比 {float(evidence.get('low_delta_t_ratio', 0)):.1%}。")
            return "\n".join(lines)
        lines = [f"Validated project summary: consistency validation {'passed' if validation else 'failed'}.", f"{len(kpis)} equipment analyzed; {len(findings)} findings; {len(self.context.opportunities)} opportunities."]
        lines += [f"- {item.equipment_name}: COP {item.metric_summary.get('cop', {}).get('mean', 0):.2f}; ΔT {item.metric_summary.get('delta_t_c', {}).get('mean', 0):.2f} °C; {'available' if item.status == 'available' else 'unavailable'}." for item in kpis]
        return "\n".join(lines)

    def _equipment_name(self, message: str) -> str | None:
        organization = getattr(self.context, "equipment_organization", None)
        candidates = [binding.equipment.name for binding in organization.heat_sources] if organization else []
        text = message.casefold()
        matched = next((name for name in candidates if name.casefold() in text), None)
        return matched or normalize_equipment_id(message)
