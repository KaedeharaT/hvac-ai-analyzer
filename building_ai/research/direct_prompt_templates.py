"""LLM Direct Baseline 的公共 Prompt 模板；本模块不执行 LLM 请求。"""

from building_ai.config.compat import DIRECT_PROMPT_VERSIONS


DIRECT_SYSTEM_PROMPT = "あなたはHVACデータの専門家です。必ずJSONで回答してください。"

DIRECT_LABELS = (
    "heat_source_supply_temp",
    "heat_source_return_temp",
    "heat_source_flow",
    "heat_source_power",
    "heat_source_energy",
    "heat_source_capacity",
    "terminal_supply_air_temp",
    "terminal_return_air_temp",
    "terminal_air_volume",
    "terminal_power",
    "terminal_energy",
    "terminal_capacity",
    "other",
)

_LABEL_LIST = "\n".join(f'- "{label}"' for label in DIRECT_LABELS)

DIRECT_LABEL_DEFINITIONS = """ラベル定義：
- heat_source_supply_temp: 熱源設備の供給・往き・出口側温度
- heat_source_return_temp: 熱源設備の還り・戻り・入口側温度
- heat_source_flow: 熱源設備の水流量
- heat_source_power: 熱源設備の瞬時消費電力
- heat_source_energy: 熱源設備の積算電力量
- heat_source_capacity: 熱源設備の冷却・加熱能力
- terminal_supply_air_temp: AHU、FCU、室内機など末端設備の給気・吹出温度
- terminal_return_air_temp: 末端設備の還気・吸込温度
- terminal_air_volume: 末端設備の風量
- terminal_power: 末端設備の瞬時消費電力
- terminal_energy: 末端設備の積算電力量
- terminal_capacity: 末端設備の冷却・加熱能力
- other: 上記に該当しない指標、状態、異常、制御、その他の列"""


def validate_direct_prompt_version(version: str) -> str:
    version = str(version or "").strip()
    if version not in DIRECT_PROMPT_VERSIONS:
        allowed = ", ".join(sorted(DIRECT_PROMPT_VERSIONS))
        raise ValueError(
            f"非法 Direct Prompt Version: {version!r}. 允许值: {allowed}"
        )
    return version


def build_direct_prompt(
    version: str,
    column_name: str,
    sample_values=None,
    stats=None,
    neighbor_cols=None,
) -> str:
    """构建 Direct Prompt；四个版本只有【入力情報】区域不同。"""
    version = validate_direct_prompt_version(version)

    input_lines = [f'列名: "{column_name}"']
    if version in {
        "direct_v2_name_values",
        "direct_v3_name_values_stats",
        "direct_v4_full_context",
    }:
        if sample_values is None:
            raise ValueError(f"{version} 需要 series/sample_values，不能退化为 V1")
        input_lines.append(f"サンプル値: {list(sample_values)}")
    if version in {"direct_v3_name_values_stats", "direct_v4_full_context"}:
        if stats is None:
            raise ValueError(f"{version} 需要 stats，不能退化为低版本")
        input_lines.append(f"基本統計量: {stats}")
    if version == "direct_v4_full_context":
        neighbors = list(neighbor_cols or [])
        input_lines.append("同じデータ表の他の列名:")
        input_lines.extend(f"- {name}" for name in neighbors)

    input_block = "\n".join(input_lines)
    return f"""あなたは建築設備・HVACデータの専門家です。
次の入力情報を使い、以下の 13 個のどの役割に最も当てはまるかを直接判断してください。

【入力情報】
{input_block}

候補タグ（tag）は次の 13 個のいずれかです：
{_LABEL_LIST}

{DIRECT_LABEL_DEFINITIONS}

注意：
- 列名に「COP」が含まれる場合、その列は性能指標であり物理量そのものではないため、必ず "other" に分類してください。
- 列名に「異常」「エラー」「コード」「予備」「保留」「dummy」などが含まれる場合も、原則 "other" としてください。
- 列名に「電力量」「積算」「日算値」「日算」「kWh」「Wh」「energy」が含まれる場合は、
  瞬時電力ではなく「積算電力量」を表す可能性が高いため、
  "heat_source_energy" または "terminal_energy" を優先的に検討してください。
- 列名に「消費電力（瞬時値）」「電力」「kW」が含まれ、
  「積算」「日算値」「電力量」などが含まれない場合は、瞬時電力（power）として扱ってください。

必ず標準 JSON だけを返してください。形式は次の通りです：
{{
  "tag": "<上の候補タグのどれか1つ>",
  "confidence": 0.0
}}

条件：
- tag は上記の 13 個のどれか「1つだけ」にしてください。
- confidence は、あなたの判断の自信度を表す 0.0〜1.0 の数値です。
- Markdown、コードブロック、追加の説明文やコメントは一切書かず、JSON のみを返してください。
"""
