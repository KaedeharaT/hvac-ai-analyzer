"""Small, attributable BuildingAI knowledge catalog.

The catalog deliberately stores original, factual summaries rather than copied
manuals, standards, or web pages.  Every item points to an entry in
``knowledge/source_registry.json`` and is suitable for local SQLite ingestion.
"""
from __future__ import annotations

from pathlib import Path
import json
import re
from collections import Counter, defaultdict

CATALOG_DIR = Path(__file__).resolve().parents[1] / "knowledge"


def source_registry() -> list[dict]:
    return json.loads((CATALOG_DIR / "source_registry.json").read_text(encoding="utf-8"))


def _cards(source_id: str, country: str, language: str, category: str, equipment: str,
           concepts: list[str], rows: list[tuple[str, str]]) -> list[dict]:
    return [
        {"record_id": f"{source_id}-{category}-{equipment}-{index:02d}", "source_id": source_id, "country": country,
         "language": language, "knowledge_category": category, "equipment_type": equipment,
         "concepts": concepts, "title": title, "section": category.replace("_", " ").title(), "text": text}
        for index, (title, text) in enumerate(rows, 1)
    ]


def curated_facts() -> list[dict]:
    facts: list[dict] = []
    facts += _cards("us_project_haystack", "US", "English", "semantic", "bems_point",
        ["bems_point", "chilled_water_supply_temperature", "semantic_mapping", "sensor"], [
        ("Haystack point containment", "Model a BEMS point with a site reference and an equipment reference so its measured or controlled context is explicit."),
        ("Haystack point function", "Classify a point as a sensor, command, or setpoint; a point should have one primary point function."),
        ("Haystack chilled-water supply temperature", "Describe chilled-water supply temperature with the concepts chilled, water, supply, temperature, point, and sensor, plus equipment and site references."),
        ("Haystack tags are formal definitions", "Haystack definitions bind a symbolic tag to a documented meaning and can organize tags into a taxonomy."),
        ("Haystack conjuncts", "A compound concept is represented by its individual marker tags; tag order in a conjunct distinguishes different concepts."),
        ("Haystack missing value", "Use an explicit unavailable or invalid value rather than silently interpreting a missing BEMS observation as a measurement."),
        ("Haystack relationship context", "Tags alone identify a concept, while references such as equipment and site references place a point in a building context."),
        ("Haystack semantic review", "When a raw BEMS name is ambiguous, preserve the original name and attach a reviewed semantic classification rather than overwriting the source."),
    ])
    facts += _cards("us_brick", "US", "English", "semantic", "hvac_equipment",
        ["brick", "equipment", "sensor", "system_topology"], [
        ("Brick equipment telemetry", "Use brick:hasPoint to connect equipment to telemetry that measures, controls, configures, or monitors that equipment."),
        ("Brick point ownership", "Use brick:isPointOf as the inverse relationship when starting from a sensor or setpoint and linking it to the equipment it describes."),
        ("Brick flow topology", "Use brick:feeds for an upstream-to-downstream flow relationship; do not use it merely to group unrelated assets."),
        ("Brick composition", "Use brick:hasPart for a physical or structural component such as a fan or coil that is part of an air-handling unit."),
        ("Brick control relationship", "Use brick:controls or brick:isControlledBy to express the relationship between a controller and the equipment it supervises."),
        ("Brick point versus collection", "A point collection organizes application-level groups but does not replace explicit point ownership, topology, or control relationships."),
        ("Brick meter model", "A meter is equipment whose associated points describe energy, power, water, gas, steam, or another measured substance."),
        ("Brick semantic boundary", "A relationship model records how entities are connected; it does not prove that a specific building currently has a fault."),
    ])
    facts += _cards("us_open223", "US", "English", "system_topology", "hvac_system",
        ["open223", "system_topology", "pipe", "duct", "chilled_water_loop"], [
        ("223 topology scope", "ASHRAE 223-style topology represents how media such as water, air, or electricity move between connected building entities."),
        ("Connection points", "Associate equipment with inlet, outlet, or bidirectional connection points when modeling where a pipe, duct, or wire attaches."),
        ("Connection media", "Record the conveyed medium on a connection point so chilled-water, air, and electrical relationships are not conflated."),
        ("Topology and direction", "A directed connection describes flow topology; it should be supported by the system model rather than inferred from an isolated point name."),
    ])
    facts += _cards("us_brick", "US", "English", "equipment", "structured_hvac_equipment",
        ["brick", "equipment", "sensor", "ahu", "fan", "pump", "chiller", "meter"], [
        ("Brick AHU representation", "Represent an air-handling unit as equipment and relate its fan, coil, dampers, sensors, commands, and setpoints with explicit Brick relationships."),
        ("Brick pump representation", "Represent a pump as equipment and connect measured power, speed, differential pressure, and command points to the pump rather than to an unrelated system."),
        ("Brick chiller representation", "Represent a chiller as equipment and associate leaving-water temperature, entering-water temperature, power, status, and control points with the chiller context."),
        ("Brick sensor semantics", "A sensor class names what a point measures; combine it with point ownership and location to avoid treating the same temperature point as interchangeable across assets."),
        ("Brick setpoint semantics", "A setpoint is a control value, not evidence that the measured process reached it; retain both setpoint and sensor relationships in the model."),
        ("Brick equipment discovery", "Explicit equipment, point, part, feed, and control relationships make equipment discovery more reliable than grouping raw point names by text alone."),
    ])
    facts += _cards("us_energyplus", "US", "English", "engineering_principles", "chiller",
        ["chiller", "cop", "part_load", "chilled_water_supply_temperature", "flow", "load"], [
        ("Chiller reference conditions", "Chiller capacity and energy models use reference temperatures and flows; compare observed performance against an appropriate operating context."),
        ("Part-load performance", "Chiller energy performance varies with part-load ratio and temperature conditions, so a single COP should be interpreted with load and operating conditions."),
        ("Leaving chilled-water temperature", "Leaving evaporator water temperature is commonly the chilled-water supply temperature and is controlled toward a setpoint when capacity is available."),
        ("Capacity before diagnosis", "If required heat transfer exceeds available chiller capacity, a temperature deviation can result without proving a mechanical fault."),
        ("Temperature-reset evaluation", "Evaluate chilled-water supply-temperature reset together with load, humidity or process requirements, comfort constraints, and chiller power."),
        ("Flow context", "Water flow affects heat transfer and temperature difference; interpret a low delta-T finding with flow, valves, bypasses, and terminal conditions."),
        ("COP interpretation", "COP is output divided by input under a defined boundary; missing load or power data means a project-specific COP should not be invented."),
        ("Trend evidence", "Use time-series trends to distinguish a persistent operating pattern from a short transient before recommending an equipment change."),
        ("Setpoint changes", "A setpoint adjustment is an operational experiment, not a universal remedy; verify capacity, comfort, and energy response after a controlled change."),
        ("Plant interaction", "Chillers, pumps, cooling towers, and controls interact; a system-level review is more reliable than attributing a plant issue to one component from one KPI."),
    ])
    facts += _cards("us_doe_femp_om", "US", "English", "operation_maintenance", "hvac_system",
        ["maintenance", "operation", "commissioning", "energy_saving", "hvac"], [
        ("O&M criteria", "Define operating and maintenance criteria for significant energy uses and use them consistently to support efficient, reliable operation."),
        ("Maintenance planning", "A maintenance schedule and checklist help make inspections repeatable and make energy, water, and indoor-environment objectives visible."),
        ("Trend before intervention", "Use BAS trend data to verify an operating issue and the result of a change instead of treating an alarm or one point value as conclusive."),
        ("Low-cost operational work", "Re-tuning and recommissioning can target operational inefficiencies through data-driven review before a capital retrofit is selected."),
        ("Equipment run hours", "Review unnecessary run hours, schedules, overrides, and simultaneous heating and cooling as practical first checks for avoidable energy use."),
        ("Maintenance evidence", "Inspect and document controls, sensors, valves, filters, heat exchangers, and rotating equipment before asserting a specific root cause."),
        ("Alarm follow-up", "An alarm is a prompt for evidence collection; verify the affected point, timeframe, equipment context, and operating consequence."),
        ("Commissioning boundary", "Operational adjustments should respect safety, comfort, process, and manufacturer requirements and receive site validation."),
        ("Measurement verification", "After an O&M measure, compare an appropriate baseline and post-change measurement period instead of claiming savings from a recommendation alone."),
        ("O&M escalation", "Escalate from no-cost checks to repair or retrofit after evidence supports the need and the operational risk has been assessed."),
    ])
    facts += _cards("us_doe_better_buildings", "US", "English", "energy_saving", "chilled_water_system",
        ["chiller", "pump", "control", "retrofit", "energy_saving", "chilled_water_loop"], [
        ("System-level cooling review", "Review chillers, cooling towers, pumps, and air-handling equipment as an interacting cooling system when seeking energy savings."),
        ("Variable-speed opportunity", "Variable-frequency drives on chilled-water pumps, condenser-water pumps, and cooling-tower fans are candidates for a site-specific savings assessment."),
        ("Compressor staging", "Appropriate compressor or chiller staging can improve system energy performance when matched to the actual load profile."),
        ("Two-way valve review", "Conversion from constant to variable flow and three-way to two-way control valves requires an engineering review of hydraulics and control stability."),
        ("Temperature reset", "Reset chilled-water supply and condenser-water entering-temperature setpoints in response to load and ambient conditions only after validating service requirements."),
        ("Free cooling", "Waterside economizer or free-cooling strategies are retrofit candidates where climate, loads, equipment, and water-side design make them feasible."),
        ("Pump high-frequency check", "Long periods of high pump speed justify checking differential-pressure setpoints, valve authority, bypass flow, and actual load before resizing equipment."),
        ("Retrofit economics", "Capital cooling-system changes require site-specific feasibility, operating hours, baseline energy, cost, and maintenance analysis."),
    ])
    facts += _cards("us_doe_better_buildings", "US", "English", "energy_saving", "hvac_measure",
        ["fan", "pump", "heat_recovery", "measurement", "retrofit", "energy_saving"], [
        ("Fan control candidate", "Variable-speed control for fans is a candidate measure when air volume can follow real demand and ventilation requirements remain satisfied."),
        ("Heat-recovery candidate", "Heat-recovery opportunities require review of climate, exhaust and outdoor-air flows, contamination risk, pressure drop, and maintenance access."),
        ("Metering for measures", "Metering and trend data establish a baseline for an energy measure and help distinguish a persistent saving from a weather or operating change."),
        ("Replacement screening", "Equipment replacement candidates should be screened with remaining life, load profile, maintenance condition, utility cost, installation constraints, and verification planning."),
    ])
    facts += _cards("us_doe_retuning", "US", "English", "operation", "building_controls",
        ["operation", "control", "energy_saving", "measurement", "night_power"], [
        ("Re-tuning principle: off", "If a system is not needed, turning it off is a first operational principle to investigate through schedules and measured demand."),
        ("Re-tuning principle: turn down", "If full output is not required, evaluate whether a controlled reduction can maintain service while reducing energy use."),
        ("Avoid simultaneous conditioning", "Investigate simultaneous heating and cooling through schedules, setpoints, valves, and control sequences before assigning fault responsibility."),
        ("Night power investigation", "High overnight power can be associated with schedule overrides, after-hours loads, plant enable logic, or unmet loads; use project trends to distinguish them."),
        ("Demand-responsive operation", "Operating criteria should reflect actual demand while preserving required indoor conditions and equipment constraints."),
        ("Operational sequence review", "Review start/stop sequences and staging logic when frequent cycling or excess run time appears in BEMS data."),
        ("No-cost first", "Data-driven re-tuning is appropriate for evaluating low- or no-cost operational opportunities before capital projects."),
        ("Evidence separation", "General O&M guidance explains what to check; only the project’s own measurements establish what is occurring at that site."),
    ])
    facts += _cards("cn_mee_public_institutions", "China", "Chinese", "operation_maintenance", "public_building",
        ["operation", "maintenance", "measurement", "energy_saving", "bems"], [
        ("分项计量与实时监测", "公共机构应按用能种类、系统实行分类分项计量并对能耗状况进行实时监测，以便及时发现和纠正浪费。"),
        ("运行管理制度", "公共机构应建立用能系统操作规程，加强设备运行调节、维护保养和巡视检查，并推广低成本、无成本节能措施。"),
        ("空调运行管理", "空调室内温度控制、自然通风利用和运行管理改进应结合建筑实际需求与相关规定执行。"),
        ("能源审计用途", "节能改造前后应通过能源审计、投资收益分析和计量考核来明确并验证节能指标。"),
        ("待机能耗", "对于夜间或非使用时段的高功率，应先核查待机、运行计划、人工覆盖和必要服务负荷。"),
        ("专业岗位", "重点用能系统和设备的操作岗位应配备专业技术人员，复杂调整不应仅凭通用问答执行。"),
        ("数据质量", "能耗统计需基于真实、准确的计量数据；缺失或异常计量不能支撑精确的项目结论。"),
        ("持续改进", "将发现的问题、采取的措施和计量结果记录在台账中，有助于形成可复核的运行优化闭环。"),
    ])
    facts += _cards("cn_ndrc_building_decarbonization", "China", "Chinese", "energy_saving", "existing_building",
        ["retrofit", "energy_saving", "operation", "maintenance", "hvac"], [
        ("运行节能重点", "建筑节能降碳工作将公共建筑节能监管、室温控制机制和重点用能设备调试保养列为运行管理重点。"),
        ("既有建筑改造", "既有建筑改造应结合建筑现状、重点用能设备和实施条件制定方案，不把通用措施直接当作确定收益。"),
        ("低效设备更新", "淘汰低效设备是改造方向之一，但替换前应评估负荷、运行小时、系统匹配、投资和验证方案。"),
        ("能源监管", "公共建筑能源监管可为能耗异常筛查和优化优先级提供数据基础，但不能代替现场诊断。"),
        ("调试保养", "对重点设备开展调试、维护和保养有助于在改造前先消除可纠正的运行问题。"),
        ("项目化评估", "改造目标、节能量和经济性应以项目测量、审计或计算为依据，而非由知识条目直接推定。"),
        ("设备系统协同", "冷热源、输配系统与末端设备应协同评价，避免只依据单一设备功率判断全系统效率。"),
        ("低成本优先", "在明确安全、舒适与工艺边界后，通常先评估运行管理和控制优化，再进行资本性改造筛选。"),
    ])
    facts += _cards("cn_ndrc_best_practices", "China", "Chinese", "control", "public_building",
        ["bems", "control", "measurement", "energy_saving", "operation"], [
        ("BEMS 与分项计量", "建筑自控系统与分项计量系统结合，可支持运行数据的采集、统计、分析和优化管理。"),
        ("空调待机能耗", "对空调待机能耗的管控应从运行时间、待机策略、控制逻辑和实际服务需求进行核查。"),
        ("运行模式", "建筑功能、使用模式和空间时段是解释能耗趋势与制定控制策略的重要上下文。"),
        ("数据驱动优化", "先从数据中识别可疑模式，再通过调节和管控措施验证，避免把相关性当成单一因果。"),
        ("集成运维", "BIM、设施管理与能耗信息可在合法、适当的数据治理下支持运维协同，但不改变现场安全责任。"),
        ("智能管控边界", "智能控制建议应由现场授权人员审查；BuildingAI 仅提供只读分析与建议依据。"),
    ])
    facts += _cards("cn_mohurd_ndrc_building", "China", "Chinese", "retrofit", "existing_public_building",
        ["retrofit", "energy_saving", "chiller", "pump", "fan", "measurement", "economic_evaluation"], [
        ("既有公共建筑改造", "既有公共建筑节能改造应结合围护结构、冷热源、输配系统、末端设备和运行管理进行系统化筛选。"),
        ("冷热源更新", "冷水机组、热泵等冷热源更新前应核查实际负荷、运行效率、维修状态和与输配系统的匹配。"),
        ("泵与风机优化", "水泵和风机优化应以实际流量、压差、阀门工况、运行时间和舒适性约束为依据。"),
        ("能源审计边界", "能源审计用于发现节能潜力和验证改造效果；不完整计量数据应作为不确定性而不是精确节能量。"),
        ("改造经济性", "设备改造候选应同时评估投资、运行费用、维护、寿命、施工影响和可测量的节能指标。"),
        ("建筑运行评估", "运行评估应覆盖季节、使用时段和主要系统负荷，避免只根据短时功率判断改造优先级。"),
        ("高效制冷行动", "高效制冷和智能管控可作为改造方向，但现场设计、调试和运行验证仍不可省略。"),
        ("改造后验证", "改造完成后应采用可比较的计量周期和运行条件考核节能指标并记录偏差原因。"),
    ])
    facts += _cards("jp_meti_zeb", "Japan", "Japanese", "zeb", "commercial_building",
        ["zeb", "energy_saving", "retrofit", "hvac", "bems"], [
        ("ZEB の基本", "ZEB は日射遮蔽、自然エネルギー利用、高断熱、高効率設備と創エネを組み合わせ、年間エネルギー消費の大幅削減を目指す建築物の考え方です。"),
        ("設計と運用", "ZEB の性能は設計技術だけでなく、運用時の負荷、制御、保全および計測の確認を要します。"),
        ("BEMS の役割", "BEMS は建築設備のエネルギー使用状況を把握し、需要に応じた運用改善を検討するための情報基盤です。"),
        ("ZEB 診断", "ZEB 化の診断・計画では、建物用途、設備、負荷特性、改修制約を確認して候補を比較します。"),
        ("高効率設備", "高効率機器の採用候補は、既存機器の実負荷、運転時間、保守性、更新時期とあわせて評価します。"),
        ("創エネとの区別", "省エネの運用改善と再生可能エネルギー導入は別の手段であり、運転データでは両者を区別して評価します。"),
        ("計測の必要性", "改修や制御変更の効果は、比較可能な運用条件と計測データで検証します。"),
        ("快適性の制約", "空調の省エネ化は、室内環境、利用者の快適性、必要なサービス水準を損なわない範囲で検討します。"),
    ])
    facts += _cards("jp_env_zeb_technology", "Japan", "Japanese", "energy_saving", "hvac_system",
        ["zeb", "hvac", "chiller", "control", "retrofit", "temperature"], [
        ("空調負荷の低減", "外皮断熱や日射遮蔽で空調負荷を抑えた上で、高効率空調と適切な制御を組み合わせることが重要です。"),
        ("中央熱源方式", "中央熱源方式では、熱源を集約し冷温水を空調機へ送るため、熱源・ポンプ・末端を系統として確認します。"),
        ("分散熱源方式", "個別分散方式では、ゾーンごとの負荷と機器の配置・制御条件を確認して運転改善を検討します。"),
        ("設定値変更", "温度設定の変更は、負荷、湿度、快適性および設備能力を確認し、小さく変更して結果を観察します。"),
        ("高効率空調", "高効率空調の効果は設備単体だけでなく、外皮性能、日射、換気、制御と運用の影響を受けます。"),
        ("温熱環境", "快適性は温湿度だけでなく、気流、放射、着衣量、代謝などにも影響されるため、電力だけで判断しません。"),
        ("既存設備の確認", "既存設備の改善では、保守状態、制御弁、ポンプ、熱交換とセンサーの状態を確認してから対策を優先付けます。"),
        ("省エネと品質", "省エネ対策は室内環境の品質を維持することと両立させる必要があります。"),
    ])
    facts += _cards("jp_env_zeb_retrofit", "Japan", "Japanese", "retrofit", "commercial_building",
        ["zeb", "retrofit", "energy_saving", "hvac", "measurement"], [
        ("設備容量の最適化", "改修時のダウンサイジングは、実際の使用状況とエネルギー消費の把握に基づき、必要容量を再評価して検討します。"),
        ("更新候補の評価", "高効率空調、照明、断熱、制御および創エネの候補は、建物ごとの負荷と改修制約を考慮して組み合わせます。"),
        ("実測の利用", "設備更新前に実測からピークと通常負荷を確認することで、過大な能力選定のリスクを低減できます。"),
        ("改修の順序", "まず運用と保全で是正できる事項を確認し、その後に更新・容量最適化の技術経済性を評価します。"),
        ("投資判断", "更新投資は初期費用だけでなく、運転費、保守、寿命、利用者への影響を含めて評価します。"),
        ("既存建物の制約", "既存改修では配管、電源、設置空間、工事期間、運用継続などの制約を早期に確認します。"),
        ("検証計画", "改修後は、事前に決めたベースラインと計測方法で効果を確認します。"),
        ("段階的改善", "大規模改修の前でも、計測、制御、保全の改善で実行可能な低コスト対策を選別できます。"),
    ])
    facts += _cards("jp_bri_becc", "Japan", "Japanese", "energy_analysis", "non_residential_building",
        ["bems", "measurement", "energy_analysis", "chiller", "pump", "fan", "zeb"], [
        ("非住宅建築物の評価", "非住宅建築物の省エネルギー性能は、用途、外皮、空調、換気、照明、給湯などの条件を区別して評価します。"),
        ("一次エネルギーの整理", "エネルギー性能の比較では、電力だけでなく一次エネルギーの評価境界と設備用途を明確にします。"),
        ("BEMS データの利用", "BEMS の計測・蓄積データは、設備別・用途別の運転傾向を確認する基盤であり、データ品質も合わせて確認します。"),
        ("空調設備の入力条件", "空調の評価では、熱源方式、搬送、制御、設定値、運転時間および建物用途の条件が結果に影響します。"),
        ("ポンプ搬送の確認", "ポンプの省エネ評価では、流量、揚程、制御方式、運転時間と系統の抵抗変化を確認します。"),
        ("ファン搬送の確認", "ファンの運転改善では、必要外気量、空気量、圧力、フィルタ状態、制御方式と室内環境を同時に確認します。"),
        ("計測の比較可能性", "変更前後を比較する時は、気象、利用状況、運転時間、負荷条件の差を記録して解釈します。"),
        ("モデルと実測", "設計時のエネルギー計算結果は実運用の代替ではないため、実測による運転確認と継続的な改善が必要です。"),
    ])
    synthesis = _cards("buildingai_engineering_synthesis", "Global", "Multilingual", "engineering_principles", "chilled_water_system",
        ["low_delta_t", "chilled_water_pump", "bypass_valve", "sensor_quality", "flow", "maintenance"], [
        ("Low delta-T: confirm the measurement", "Before changing plant controls, compare supply and return temperature sensors, timestamps, units, and plausible readings; a sensor error can mimic low delta-T."),
        ("Low delta-T: check flow context", "A low chilled-water temperature difference can be consistent with more flow than the active load requires. Review pump speed, differential-pressure control, and valve position with project trends."),
        ("Low delta-T: check bypass path", "Inspect bypass control and unintended bypass flow because water that bypasses terminal heat exchange can reduce observed system delta-T."),
        ("Low delta-T: check terminal heat exchange", "Review coil, valve, air-side or load-side conditions when chilled water returns with little temperature rise; this is a check list, not a diagnosis."),
        ("Low delta-T: use a safe trial", "After evidence review and site approval, make a small, reversible control adjustment and observe delta-T, comfort, load, and power together."),
        ("Frequent cycling review", "Frequent starts and stops warrant review of load signal quality, staging thresholds, minimum on/off timers, enable logic, and equipment constraints."),
        ("High night power review", "When night power stays high, compare schedules, overrides, occupancy or process demand, plant enable signals, and equipment run status before recommending shutdown."),
        ("Recommendation boundary", "This checklist suggests evidence to collect. It does not establish a project condition without formal BuildingAI project data and diagnostics."),
    ])
    for fact in synthesis[:5]:
        fact["concepts"] = ["low_delta_t", "chilled_water_pump", "bypass_valve", "flow", "sensor_quality"]
    synthesis[5]["concepts"] = ["short_cycling", "operation", "maintenance"]
    synthesis[6]["concepts"] = ["night_energy", "operation", "energy_management"]
    synthesis[7]["concepts"] = ["operation", "maintenance", "energy_saving"]
    facts += synthesis
    # Structured multilingual concept records are useful to semantic mapping and
    # cross-language retrieval, rather than being translated copies of documents.
    concept_rows = [
        ("chilled_water_supply_temperature", "Normalized concept: chilled-water supply temperature. Aliases: 冷冻水供水温度, 冷水供給温度, 冷水温度, chilled water supply temperature."),
        ("chilled_water_return_temperature", "Normalized concept: chilled-water return temperature. Aliases: 冷冻水回水温度, 冷水還り温度, chilled water return temperature."),
        # Controlled natural-language variants are concept metadata, not
        # per-question routing rules.  They keep equivalent CJK and English
        # phrasings discoverable through the same normalized concept.
        ("low_delta_t", "Normalized concept: low chilled-water delta-T. Aliases: 冷冻水温差低, 温差偏低, 冷水温度差が小さい, 冷水温度差が低い, low delta T, chilled-water delta-T, chilled water delta-T."),
        ("chilled_water_pump", "Normalized concept: chilled-water pump. Aliases: 冷冻水泵, 冷水ポンプ, chilled water pump."),
        ("chiller", "Normalized concept: chiller. Aliases: 冷水机组, 冷凍機, チラー, chiller."),
        ("heat_pump", "Normalized concept: heat pump. Aliases: 热泵, ヒートポンプ, heat pump."),
        ("coefficient_of_performance", "Normalized concept: coefficient of performance. Aliases: COP, 能效比, 成績係数."),
        ("part_load", "Normalized concept: part load. Aliases: 部分负荷, 部分負荷, part load."),
        ("variable_frequency_drive", "Normalized concept: variable-frequency drive. Aliases: 变频器, インバータ, VFD."),
        ("bypass_valve", "Normalized concept: bypass valve. Aliases: 旁通阀, バイパス弁, bypass valve."),
        ("energy_management", "Normalized concept: building energy management. Aliases: 建筑能源管理, BEMS, ビルエネルギー管理."),
        ("equipment_relationship", "Normalized concept: equipment/point relationship. Aliases: 设备和传感器关系, 設備とセンサーの関係, equipment point relationship."),
        ("night_energy", "Normalized concept: high night energy. Aliases: 夜间高功率, 夜間の高い電力, overnight power."),
        ("short_cycling", "Normalized concept: frequent start-stop. Aliases: 频繁启停, 頻繁な起動停止, short cycling."),
        ("maintenance", "Normalized concept: operation and maintenance. Aliases: 运维, 保全, maintenance."),
        ("retrofit", "Normalized concept: energy retrofit. Aliases: 节能改造, 省エネ改修, energy retrofit."),
        ("sensor_quality", "Normalized concept: sensor and measurement quality. Aliases: 传感器异常, センサー品質, sensor quality."),
        ("air_handling_unit", "Normalized concept: air handling unit. Aliases: 空气处理机组, 空調機, AHU."),
    ]
    structured = _cards("us_project_haystack", "Global", "Multilingual", "semantic", "concept_dictionary",
        ["semantic_mapping", "bems_point", "equipment"], concept_rows)
    for fact in structured:
        fact["concepts"] = [fact["title"]]
    facts += structured
    # EnergyPlus's flow-context guidance is also a public, attributable
    # explanation source for a low-ΔT investigation.
    for fact in facts:
        if fact["source_id"] == "us_energyplus" and fact["title"] == "Flow context":
            fact["concepts"] = [*fact["concepts"], "low_delta_t", "bypass_valve"]
    return facts


def _terms(text: str) -> set[str]:
    normalized = text.casefold()
    values = set(re.findall(r"[\w-]+", normalized))
    for span in re.findall(r"[\u4e00-\u9fffぁ-んァ-ンー]+", normalized):
        values.add(span)
        values.update(span[index:index + 2] for index in range(max(0, len(span) - 1)))
        values.update(span[index:index + 3] for index in range(max(0, len(span) - 2)))
    return {value for value in values if len(value) > 1}


def materialize_catalog(destination: Path = CATALOG_DIR) -> dict:
    """Write deterministic, Git-safe curated content, chunks, metadata and index.

    The files contain original summaries and structured ontology facts only; no
    downloaded HTML, raw manuals, user data, or copyright-unclear full text.
    """
    registry = source_registry()
    sources = {item["source_id"]: item for item in registry}
    entries = []
    for fact in curated_facts():
        source = sources[fact["source_id"]]
        entries.append({
            "chunk_id": f"catalog:{fact['record_id']}", "source_id": fact["source_id"], "country": fact["country"],
            "language": fact["language"], "organization": source["organization"], "title": fact["title"],
            "section": fact["section"], "category": fact["knowledge_category"], "equipment_type": fact["equipment_type"],
            "concepts": fact["concepts"], "content": fact["text"], "source_url": source["official_url"],
            "license_note": source["license_or_usage_note"], "content_strategy": source["content_strategy"],
            "citation": f"{source['organization']} — {source['title']} — {fact['section']}",
        })
    entries.sort(key=lambda item: item["chunk_id"])

    def write_jsonl(path: Path, values: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in values), encoding="utf-8")

    write_jsonl(destination / "chunks" / "knowledge_chunks.jsonl", entries)
    names = {"China": "china", "US": "us", "Japan": "japan", "Global": "global"}
    for country, folder in names.items():
        write_jsonl(destination / "curated" / folder / "knowledge.jsonl", [item for item in entries if item["country"] == country])

    concept_rows = [item for item in entries if item["equipment_type"] == "concept_dictionary"]
    metadata_dir = destination / "metadata"; metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.joinpath("concept_aliases.json").write_text(json.dumps(concept_rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    inverted: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        for term in _terms(" ".join((entry["title"], entry["section"], entry["content"], " ".join(entry["concepts"])))):
            inverted[term].append(entry["chunk_id"])
    index_dir = destination / "index"; index_dir.mkdir(parents=True, exist_ok=True)
    index_dir.joinpath("keyword_cjk_index.json").write_text(
        json.dumps({term: sorted(ids) for term, ids in sorted(inverted.items())}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1, "source_count": len(registry), "chunk_count": len(entries), "index_term_count": len(inverted),
        "chunks_by_country": dict(Counter(item["country"] for item in entries)),
        "chunks_by_language": dict(Counter(item["language"] for item in entries)),
        "chunks_by_category": dict(Counter(item["category"] for item in entries)),
        "content_policy": "Original factual summaries, structured ontology facts, and source metadata only.",
    }
    metadata_dir.joinpath("catalog_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_materialized_facts(destination: Path = CATALOG_DIR) -> list[dict]:
    """Load the versioned JSONL content used by the runtime database builder."""
    path = destination / "chunks" / "knowledge_chunks.jsonl"
    return [
        {"record_id": item["chunk_id"].removeprefix("catalog:"), "source_id": item["source_id"], "country": item["country"],
         "language": item["language"], "knowledge_category": item["category"], "equipment_type": item["equipment_type"],
         "concepts": item["concepts"], "title": item["title"], "section": item["section"], "text": item["content"]}
        for item in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    ]
