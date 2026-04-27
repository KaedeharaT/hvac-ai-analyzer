import os
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
# === C1–C8 Multi-slot scorer (lightweight, unit-aware) ===
import math
from collections import defaultdict

# ================================================================
# 统一的 LLM 打分器：每一列只调用一次 Qwen，返回 C1–C8 全部信息
# ================================================================

import requests, json
_SLOT_LLM_CACHE = {}   # 防止同一列重复调用 LLM




def qwen_chat_json(prompt: str):
    """调用本地 Ollama（Qwen2.5-7B）并确保返回 JSON"""
    payload = {
        "model": "qwen2.5:7b",
        "messages": [
            {"role": "system", "content": "あなたはHVACデータの専門家です。必ずJSONで回答してください。"},
            {"role": "user", "content": prompt},
        ],
        "format": "json",
        "stream": False
    }
    resp = requests.post("http://localhost:11434/api/chat", json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()["message"]["content"]

def _robust_sample_values(s: pd.Series, k=80, prefer_nonzero=True, seed=0):
    """
    从整列中抽样，避免“前40全是0”的问题：
    - 尽量从全序列均匀取样
    - 若 prefer_nonzero=True，会优先塞一些非0样本（但仍保留少量0）
    """
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return []

    arr = s.to_numpy(dtype=float)
    n = len(arr)

    # 1) 先均匀取样（覆盖全时段）
    idx = np.linspace(0, n - 1, num=min(k, n), dtype=int)
    base = arr[idx].tolist()

    if not prefer_nonzero:
        return [float(x) for x in base[:k]]

    # 2) 再补充非0样本（如果均匀采样里非0太少）
    nz = arr[np.abs(arr) > 1e-12]
    if nz.size == 0:
        return [float(x) for x in base[:k]]

    # 控制“非0样本比例”：比如 70% 非0 + 30% 原始覆盖样本
    target_nz = int(k * 0.7)
    target_base = k - target_nz

    # base里挑前 target_base
    base_part = base[:min(target_base, len(base))]

    # 非0里随机挑 target_nz（但不重复太纠结）
    rng = np.random.default_rng(seed)
    take = min(target_nz, nz.size)
    nz_part = rng.choice(nz, size=take, replace=False).tolist()

    merged = base_part + nz_part
    merged = merged[:k]
    return [float(x) for x in merged]


def compute_shape_features(series: pd.Series) -> dict:
    """
    从整列计算“形态特征”，用于告诉 LLM：这列像不像 temp/flow/power/energy/control
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {
            "count": 0,
            "zero_ratio": None,
            "nonzero_ratio": None,
            "unique_ratio": None,
            "min": None, "max": None,
            "p01": None, "p05": None, "p50": None, "p95": None, "p99": None,
            "std": None,
            "monotonic_increase_ratio": None,
            "step_change_ratio": None,
            "max_run_zero": None,
            "max_run_nonzero": None,
            "is_all_zero": None,
        }

    x = s.to_numpy(dtype=float)
    n = len(x)

    eps = 1e-12
    is_zero = np.abs(x) <= eps
    zero_ratio = float(is_zero.mean())
    nonzero_ratio = 1.0 - zero_ratio

    # unique_ratio：对“常数列/状态列”很敏感
    unique_ratio = float(pd.Series(x).nunique(dropna=True) / max(1, n))

    # 分位数比 min/max 稳（去掉极端值干扰）
    p01, p05, p50, p95, p99 = [float(v) for v in np.nanpercentile(x, [1, 5, 50, 95, 99])]
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    std = float(np.nanstd(x))

    # 单调性：energy 的强特征（不要求严格单调，允许少量回退）
    if n >= 3:
        dx = np.diff(x)
        monotonic_increase_ratio = float((dx >= -1e-9).mean())
    else:
        monotonic_increase_ratio = 0.0

    # “跳变比例”：power 的典型特征（0/某值来回跳）
    if n >= 3:
        dx = np.abs(np.diff(x))
        step_thr = max(1e-6, (p95 - p05) * 0.2)  # 相对阈值
        step_change_ratio = float((dx > step_thr).mean())
    else:
        step_change_ratio = 0.0

    # run-length：最长连续 0 段/非0段，判断“设备长时间停机”/“状态列”
    def _max_run(mask: np.ndarray) -> int:
        # mask: bool array
        max_run = run = 0
        for v in mask:
            if v:
                run += 1
                if run > max_run:
                    max_run = run
            else:
                run = 0
        return int(max_run)

    max_run_zero = _max_run(is_zero)
    max_run_nonzero = _max_run(~is_zero)

    is_all_zero = bool(nonzero_ratio < 1e-6)

    return {
        "count": int(n),
        "zero_ratio": zero_ratio,
        "nonzero_ratio": nonzero_ratio,
        "unique_ratio": unique_ratio,
        "min": xmin, "max": xmax,
        "p01": p01, "p05": p05, "p50": p50, "p95": p95, "p99": p99,
        "std": std,
        "monotonic_increase_ratio": monotonic_increase_ratio,
        "step_change_ratio": step_change_ratio,
        "max_run_zero": max_run_zero,
        "max_run_nonzero": max_run_nonzero,
        "is_all_zero": is_all_zero,
    }

def _clamp01(x, default=0.0):
    try:
        v = float(x)
    except Exception:
        return float(default)
    if math.isnan(v) or math.isinf(v):
        return float(default)
    return max(0.0, min(1.0, v))

def normalize_llm_slots(obj: dict) -> dict:
    """
    强制把 LLM 输出修成你需要的 schema，并把所有分数 clamp 到 [0,1]。
    """
    if not isinstance(obj, dict):
        obj = {}

    def get_dict(k, default):
        v = obj.get(k)
        return v if isinstance(v, dict) else default

    C1 = get_dict("C1", {"heat_source":0.0,"terminal":0.0,"fan":0.0})
    C2 = get_dict("C2", {"temp":0.0,"flow":0.0,"power":0.0,"energy":0.0,"pressure":0.0})
    C3 = get_dict("C3", {"supply":0.0,"ret":0.0})
    C4 = get_dict("C4", {"variability":0.0})
    C5 = get_dict("C5", {"bundle":0.0})
    C6 = get_dict("C6", {"unit":None,"unit_type":"unknown","confidence":0.0})
    C7 = get_dict("C7", {"tag":None,"confidence":0.0})
    C8 = get_dict("C8", {"ok":0.0})

    C1 = {k: _clamp01(C1.get(k, 0.0)) for k in ["heat_source","terminal","fan"]}
    C2 = {k: _clamp01(C2.get(k, 0.0)) for k in ["temp","flow","power","energy","pressure"]}
    C3 = {k: _clamp01(C3.get(k, 0.0)) for k in ["supply","ret"]}
    C4 = {"variability": _clamp01(C4.get("variability", 0.0))}
    C5 = {"bundle": _clamp01(C5.get("bundle", 0.0))}

    # C6
    unit = C6.get("unit")
    if unit is not None:
        unit = str(unit).strip() or None
    unit_type = str(C6.get("unit_type") or "unknown").strip().lower()
    if unit_type not in ("temp","flow","power","energy","pressure","control","other","unknown"):
        unit_type = "unknown"
    C6 = {"unit": unit, "unit_type": unit_type, "confidence": _clamp01(C6.get("confidence", 0.0))}

    # C7
    tag = C7.get("tag")
    tag = str(tag).strip() if tag is not None else None
    C7 = {"tag": tag, "confidence": _clamp01(C7.get("confidence", 0.0))}

    C8 = {"ok": _clamp01(C8.get("ok", 0.0))}

    return {"C1":C1,"C2":C2,"C3":C3,"C4":C4,"C5":C5,"C6":C6,"C7":C7,"C8":C8}


def llm_score_all_slots(col_name, series, neighbor_cols=None):
    """
    调用本地 Qwen，为某一列一次性生成 C1~C8 所有槽位的评分。
    C1~C8 的语义和打分规则，都在 prompt 里详细说明。
    """
    sf = compute_shape_features(series)
    neighbor_cols = neighbor_cols or []

    key = (
        str(col_name),
        hash(tuple(neighbor_cols)),
        int(sf.get("count") or 0),
        round(float(sf.get("zero_ratio") or 0.0), 3),
        round(float(sf.get("p50") or 0.0), 3),
        round(float(sf.get("monotonic_increase_ratio") or 0.0), 3),
    )
    if key in _SLOT_LLM_CACHE:
        return _SLOT_LLM_CACHE[key]

    s = pd.to_numeric(series, errors="coerce")

    # ✅ 更稳健：从整列抽样（覆盖全时段，且优先带一些非0）
    sample = _robust_sample_values(s, k=80, prefer_nonzero=True, seed=0)

    # ✅ 基础统计保留，但主要靠 shape_features
    stats = dict(
        min=float(s.min(skipna=True)) if s.notna().any() else None,
        max=float(s.max(skipna=True)) if s.notna().any() else None,
        mean=float(s.mean(skipna=True)) if s.notna().any() else None,
        std=float(s.std(skipna=True)) if s.notna().any() else None,
    )

    # ✅ 新增：整列形态特征（你之前提的“形态特征”现在真的会进入 prompt）
    shape_features = compute_shape_features(s)

    neighbor_cols = neighbor_cols or []

    prompt = f"""
你是一名暖通空调（HVAC）データ専門家です。
与えられた「1つの列」について、以下の C1〜C8 のスロットを 0〜1 のスコアで評価し、
必ず JSON 形式で返してください。

【入力情報】
- 列名: {col_name}
- 値サンプル（最大80件）: {sample}
- 統計情報: {stats}
- 形態特徴（整列から計算）: {shape_features}
- 同じシートに存在する他の列名リスト: {neighbor_cols}

========================
★ 各スロットの定義と出力形式
========================

全体の出力は次のような JSON オブジェクトです:
{{
  "C1": {{ "heat_source": 0.0, "terminal": 0.0, "fan": 0.0 }},
  "C2": {{ "temp": 0.0, "flow": 0.0, "power": 0.0, "energy": 0.0, "pressure": 0.0 }},
  "C3": {{ "supply": 0.0, "ret": 0.0 }},
  "C4": {{ "variability": 0.0 }},
  "C5": {{ "bundle": 0.0 }},
  "C6": {{ "unit": null, "unit_type": "unknown", "confidence": 0.0 }},
  "C7": {{ "tag": null, "confidence": 0.0 }},
  "C8": {{ "ok": 0.0 }}
}}

※ すべての C1〜C8 キーを必ず含めてください。

------------------------------------------------
C1: Component type（熱源 / 末端 / ファン）
------------------------------------------------
目的：この列がどの機器グループに属するかを推定する。

- "heat_source": 主機 / 熱源ユニット（例: ヒートポンプ, 冷凍機, ボイラー）に関係する列なら高いスコア。
  - 典型的なキーワード: "AHP", "HP", "chiller", "冷凍機", "熱源", "冷温水機", "主機", "室外機" など。
- "terminal": 室内機・末端空調機（AHU, FCU, 室内機など）に関係する列なら高いスコア。
  - キーワード: "AHU", "FCU", "室内機", "fan coil", "吹出温度", "室温", "zone", "VAV" など。
- "fan": 送風機単体（ファンモータ, 給気ファン, 排気ファン）に関係する列なら高いスコア。
  - キーワード: "fan", "送風機", "風機", "blower", "EF", "SF" など。

注意：
- どれにも当てはまらない場合は、すべて 0.0 に近い値でよい。
- heat_source / terminal / fan のいずれかが 0.6〜1.0 程度で最も高くなるように意識してください。

------------------------------------------------
C2: 物理量タイプ（temp / flow / power / energy / pressure）
------------------------------------------------
目的：この列が何の物理量かを推定する。

- "temp": 温度
  - キーワード: "温度", "℃", "°C", "temp", "temperature"
  - 値の分布が 5〜40 程度でゆっくり変化する場合。
- "flow": 流量 / 風量
  - キーワード: "流量", "風量", "m3/h", "m³/h", "L/s", "L/min", "air volume"
- "power": 瞬時電力
  - キーワード: "電力", "kW", "W", "消費電力（瞬時値）"
  - 値が 0 の近くと数kW〜数十kWを行き来する場合。
- "energy": 積算電力量
  - キーワード: "電力量", "積算", "kWh", "Wh", "energy", "積算値"
  - 時間とともに単調増加（またはほぼ増加）する形。
- "pressure": 圧力
  - キーワード: "圧力", "kPa", "Pa", "MPa", "高圧", "低圧"

重要：
- 圧力や制御信号（Hz, %, V など）は、temp / power / flow / energy とは必ず区別してください。
- 例えば「高圧圧力」は "pressure" を高くし、"temp" や "power" を 0〜0.1 に抑えてください。
追加ルール（重要）:
- shape_features.is_all_zero が true、または shape_features.zero_ratio が 0.95 以上の場合、
  「機器が停止している可能性が高い」ため、C2 で power/temp/flow/energy を高くしないでください。
  この場合は unit/列名キーワードが強い時のみ例外的に高くしてよい。
- energy は shape_features.monotonic_increase_ratio が高い（例: 0.95 以上）場合にのみ高くしてください。
- power は shape_features.step_change_ratio がある程度高い（例: 0.3 以上）または列名/単位に kW/W が明確な時に高くしてください。

出力例:
"C2": {{ "temp":0.9, "flow":0.0, "power":0.1, "energy":0.0, "pressure":0.0 }}

------------------------------------------------
C3: supply / ret（供給 or 還り）
------------------------------------------------
目的：この列が供給/supply 方向か，還り/return 方向かを推定する。

- "supply" を高くする条件：
  - キーワード: "供給", "往", "出", "吹", "送", "出口", "supply", "discharge", "flow out" など。
- "ret" を高くする条件：
  - キーワード: "還", "回", "吸", "入口", "return", "inlet", "flow in" など。
- 明確に分からない場合はどちらも 0.0〜0.3 程度の低い値にしてください。

出力例:
"C3": {{ "supply":0.8, "ret":0.1 }}

------------------------------------------------
C4: variability（時系列変動の大きさ）
------------------------------------------------
目的：この列の時間変動の大きさを 0〜1 で評価する。

- stats["std"] が非常に小さい（ほぼ定数）→ variability ≈ 0.0
- ある程度ゆらぎがある通常の温度・流量 → variability ≈ 0.4〜0.7
- 急激に ON/OFF するような列（0 と大きい値を激しく行き来） → variability ≈ 0.8〜1.0

標準偏差 std の目安（完全に厳密でなくてよい）：
- std ≈ 0 → 0.0
- std が中くらい → 0.5 前後
- std が非常に大きい → 0.9 前後

出力例:
"C4": {{ "variability":0.6 }}

------------------------------------------------
C5: bundle（一緒に使われる物理グループへの適合度）
------------------------------------------------
目的：この列が、同じ機器の「供給温度・還り温度・流量・電力・能力」などの
典型的なグループに属していそうかをざっくり評価する。

ヒント：
- neighbor_cols に、同じ AHP 番号や同じ AHU 名称が付いている列が複数あれば、
  それらは同じ「バンドル（束）」を構成する可能性が高い。
- 例えば「AHP#-3-1 還水温度」「AHP#-3-1 供給水温度」「AHP#-3-1 流量」「AHP#-3-1 消費電力」など。

ルールの例：
- 自分の列名と同じ機器ID（例: "AHP#-3-1"）を含む列が neighbor_cols に複数あれば、bundle を高くする（0.6〜1.0）。
- どの列とも関係なさそうな孤立列 → bundle ≈ 0.0〜0.2。

出力例:
"C5": {{ "bundle":0.8 }}

------------------------------------------------
C6: unit / unit_type / confidence（単位情報）
------------------------------------------------
目的：この列の単位と、その「種類」を推定する。

出力形式:
"C6": {{
  "unit": "生の単位文字列。例: \"kW\", \"kWh\", \"m3/h\", \"℃\", \"kPa\", \"%\", \"Hz\" など。分からなければ null。",
  "unit_type": "temp / flow / power / energy / pressure / control / other のいずれか1つ",
  "confidence": 0.0〜1.0
}}

unit_type の定義：
- "temp": 温度に対応する単位（℃, degC, K など）
- "flow": 流量・風量（m3/h, L/s など）
- "power": 瞬時電力（kW, W など）
- "energy": 積算電力量（kWh, Wh など）
- "pressure": 圧力（Pa, kPa, MPa など）
- "control": 制御・信号（Hz, %, V, A などの制御量・指令値）
- "other": 上記以外や単位不明

重要：
- 圧力や制御信号（Hz, %, V など）は、COP/負荷率の計算とは直接関係がないので、
  それらは unit_type="pressure" や "control" として分類し、
  "temp","power","flow","energy" と必ず区別すること。
- 単位が全く分からない場合でも、unit_type を "other" として confidence を低め（0.0〜0.3）にしてください。

------------------------------------------------
C7: tag（列役割ラベル） + confidence
------------------------------------------------
目的：この列の役割を、以下の 11 クラスのいずれかとして推定する。

候補タグ（tag）：
- "heat_source_supply_temp"
- "heat_source_return_temp"
- "heat_source_flow"
- "heat_source_power"
- "heat_source_capacity"
- "terminal_supply_air_temp"
- "terminal_return_air_temp"
- "terminal_air_volume"
- "terminal_power"
- "terminal_capacity"
- "other"

ルール：
- 列名と C1/C2/C3/C6 の情報を総合して、最もありそうな 1 つの tag を選ぶ。
- かなり自信がある場合: confidence 0.8〜1.0
- どれか分からないが「other」が妥当: tag="other", confidence 0.5〜0.7
- 本当に何も分からない: tag="other", confidence 0.0〜0.3

出力例:
"C7": {{ "tag": "heat_source_return_temp", "confidence": 0.86 }}

------------------------------------------------
C8: ok（物理的妥当性 / 品質）
------------------------------------------------
目的：この列のデータが物理的に見て妥当かどうかを 0〜1 で評価する。

ヒント：
- 温度（C2.temp が高い場合）のとき：
  - [-40℃, 120℃] の範囲内でゆるやかに変化 → ok ≈ 0.8〜1.0
  - 数値が 0 または極端な値（例: 9999, -999）ばかり → ok ≈ 0.0〜0.2
- 電力・流量のとき：
  - 負の値が大量にある、または物理的にありえない桁の値 → ok を低くする
- 積算電力量のとき：
  - 時間とともに大きくなっていく傾向なら ok を高めにする。

出力例:
"C8": {{ "ok": 0.9 }}

========================
出力フォーマットの要請
========================
- 必ず上記の "C1"〜"C8" をすべて含む JSON オブジェクトを返してください。
- 各スコアは 0.0〜1.0 の範囲にしてください。
- 追加の文章や説明は書かず、JSON だけを返してください。
"""

    try:
        raw = qwen_chat_json(prompt)
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        data = normalize_llm_slots(parsed)
    except Exception as e:
        print("[LLM error]", col_name, e)
        data = normalize_llm_slots({})

    _SLOT_LLM_CACHE[key] = data
    return data
# =========================================================
# 我们要打分的 10 个标签（全局常量）
# =========================================================
SLOT_LABELS = [
    "heat_source_supply_temp",
    "heat_source_return_temp",
    "heat_source_flow",
    "heat_source_power",
    "heat_source_energy",        # ★新增：热源电力量（積算 / 日算値）
    "heat_source_capacity",

    "terminal_supply_air_temp",
    "terminal_return_air_temp",
    "terminal_air_volume",
    "terminal_power",            # ★fan_power → terminal_power
    "terminal_energy",           # ★新增：末端电力量（積算 / 日算値）
    "terminal_capacity",
]

ROLE_KEYWORDS = {
    "heat_source_supply_temp":  {"need_temp": True,  "need_supply": True,  "is_terminal": False},
    "heat_source_return_temp":  {"need_temp": True,  "need_supply": False, "is_terminal": False},
    "heat_source_flow":         {"need_flow": True,  "is_terminal": False},
    "heat_source_power":        {"need_power": True, "is_terminal": False},
    "heat_source_energy":       {"need_energy": True,"is_terminal": False},   # ★
    "heat_source_capacity":     {"need_capacity": True, "is_terminal": False},

    "terminal_supply_air_temp": {"need_temp": True,  "need_supply": True,  "is_terminal": True},
    "terminal_return_air_temp": {"need_temp": True,  "need_supply": False, "is_terminal": True},
    "terminal_air_volume":      {"need_flow": True,  "is_terminal": True},
    "terminal_power":           {"need_power": True, "is_terminal": True},    # ★
    "terminal_energy":          {"need_energy": True,"is_terminal": True},    # ★
    "terminal_capacity":        {"need_capacity": True, "is_terminal": True},
}

def _bool(val): return 1.0 if val else 0.0

def _role_device_type(role: str) -> str:
    """
    将角色映射为设备类型:
      - "heat_source" : 热源侧（主机、冷冻机、AHP 等）
      - "terminal"    : 末端侧（AHU/FCU/室内机/风机等）
      - "other"       : 无法判断或不关心
    """
    if not role or str(role).lower() == "other":
        return "other"

    meta = ROLE_KEYWORDS.get(role) or {}

    # 1) 优先用 ROLE_KEYWORDS 里的 is_terminal 标记
    if isinstance(meta, dict):
        if meta.get("is_terminal") is True:
            return "terminal"
        if meta.get("is_terminal") is False:
            return "heat_source"

    # 2) 兜底：从字符串猜
    r = (role or "").lower()
    if r.startswith("terminal_"):
        return "terminal"
    if r.startswith("heat_source_"):
        return "heat_source"

    # 不确定就放 other，不参与冲突判定
    return "other"

# ========== 可配置关键词（可选） ==========
TEMP_KEYWORDS_BASE = ["temp", "temperature", "温度", "℃", "°c"]

def _load_extra_temp_keywords(path="slot_keywords.json"):
    """
    可选：从 JSON 中加载额外的温度关键词。
    JSON 格式示例:
      {
        "temp_keywords": ["室温", "水温", "冷水温", "hot water temp"]
      }
    """
    if os.path.exists(path):
        try:
            data = json.load(open(path, "r", encoding="utf-8"))
            extra = data.get("temp_keywords", [])
            # 过滤掉空字符串
            return [k for k in extra if isinstance(k, str) and k.strip()]
        except Exception:
            pass
    return []

# =========================================================
# 正则工具：用于 C2 / C3 / C5 等槽位的启发式规则
# =========================================================
_KW   = re.compile(r"\bkw\b", re.I)
_KWH  = re.compile(r"\bkwh\b|\bwh\b|\benergy\b|積算|電力量", re.I)
_EXTRA_TEMP_KEYWORDS = _load_extra_temp_keywords()
_TEMP_TOKEN = re.compile(
    "|".join([re.escape(k) for k in (TEMP_KEYWORDS_BASE + _EXTRA_TEMP_KEYWORDS)]),
    re.I
)

_FLOW_TOKEN = re.compile(r"flow|air volume|風量|m3/h|m³/h|l/s|l/min", re.I)
_PRESSURE_TOKEN = re.compile(r"pa|kpa|mpa|pressure|圧力", re.I)
_SUPPLY_TOKEN   = re.compile(r"supply|往|出|吹|送|供|出口|outlet|flow out", re.I)
_RETURN_TOKEN   = re.compile(r"return|還|回|吸|入口|inlet|flow in", re.I)
_FAN_TOKEN      = re.compile(r"fan|送風機|風機|blower", re.I)
_CAP_TOKEN      = re.compile(r"capacity|能力|容量", re.I)
_INV_CTRL       = re.compile(r"inv|hz|%|開度|freq|frequency|set|sp|設定|status|alarm|command|code|指令", re.I)
# ===== 新增：强规则 token（直接挡住 CSV 里最危险的误判） =====
_FAULT_TOKEN = re.compile(r"異常|エラー|error|fault|alarm|警報|状態|status|コード|code|予備|保留|dummy", re.I)
_OAT_TOKEN   = re.compile(r"外気|外氣|外気温度|外氣温度|OAT|ambient", re.I)  # outdoor air temp
_DAILY_TOKEN = re.compile(r"日算値|日算|日積算|積算|電力量|kwh|wh|\benergy\b", re.I)
_FLOW_CMD_TOKEN = re.compile(r"流量指令|ポンプ流量指令|風量指令", re.I)
_FLOW_HARD_TOKEN = re.compile(r"流量指令|ポンプ流量指令|流量|風量|m3/h|m³/h|l/s|l/min|air volume", re.I)



# （可选）如果你要对 TIME 直接全 0，就加这个
_TIME_TOKEN = re.compile(r"\btime\b|日時|時間|時刻", re.I)



def is_temp_like_column(name: str,
                        c2: dict | None = None,
                        unit_type: str | None = None) -> bool:
    """
    综合三路信号判断：这个列是不是“温度类”列：
      1) 表头关键词（TEMP_TOKEN）
      2) C6 的 unit_type == "temp"
      3) C2 里 temp 分数最高且较高

    这样即使表头叫奇怪名字，只要 LLM 在 C2/C6 上判断为 temp，
    也会被认为是温度列，不用你手动不断加词。
    """
    name = str(name)

    # 1) 表头出现典型温度关键词
    if _TEMP_TOKEN.search(name):
        return True

    # 2) 单位类型直接给出 temp
    if (unit_type or "").lower() == "temp":
        return True

    # 3) C2 槽位：LLM 认为是 temp，而且置信度够高
    if isinstance(c2, dict):
        try:
            temp_score   = float(c2.get("temp", 0.0)   or 0.0)
            flow_score   = float(c2.get("flow", 0.0)   or 0.0)
            power_score  = float(c2.get("power", 0.0)  or 0.0)
            energy_score = float(c2.get("energy", 0.0) or 0.0)

            c2_scores = {
                "temp": temp_score,
                "flow": flow_score,
                "power": power_score,
                "energy": energy_score,
            }
            main_type = max(c2_scores, key=lambda k: c2_scores[k])
            main_val  = c2_scores[main_type]

            if main_type == "temp" and main_val >= 0.7:
                return True
        except Exception:
            pass

    return False



def _norm01(x, lo, hi):
    if x is None or math.isnan(x):
        return 0.0
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))

def _std01(series):
    try:
        s = pd.to_numeric(series, errors="coerce")
        sd = float(s.std(skipna=True)) if s.size else 0.0
        # heuristic: 0..5°C ~ low, >20 large swings
        return _norm01(sd, 0.0, 20.0)
    except Exception:
        return 0.0

def _range_hint(series):
    """Return rough range stats for later heuristics."""
    try:
        s = pd.to_numeric(series, errors="coerce")
        return float(s.min(skipna=True)), float(s.max(skipna=True))
    except Exception:
        return (float("nan"), float("nan"))

def _unit_from_header(h):
    """Parse likely unit from header tokens."""
    h = str(h)
    if "°c" in h.lower() or "℃" in h or "degc" in h.lower():
        return "C"
    if "kwh" in h.lower() or "wh" in h.lower():
        return "kWh"
    if "kw" in h.lower():
        return "kW"
    if "m3/h" in h.lower() or "m³/h" in h.lower():
        return "m3/h"
    if "l/s" in h.lower():
        return "L/s"
    if "l/min" in h.lower():
        return "L/min"
    if "pa" in h.lower():
        return "Pa"
    if "%" in h:
        return "%"
    return None

# -----------------------
# C1：组件类型（LLM）
# -----------------------
def c1_component_score(col, series=None, slot_llm=None):
    if callable(slot_llm):
        return slot_llm(col, series)["C1"]
    return {"heat_source":0.0,"terminal":0.0,"fan":0.0}

# -----------------------
# C2：物理量类型（LLM）
# -----------------------
def c2_physical_quantity_score(col, series, slot_llm=None):
    if callable(slot_llm):
        return slot_llm(col, series)["C2"]
    return {"temp":0.0,"flow":0.0,"power":0.0,"energy":0.0,"pressure":0.0}

def c3_relation_score(col, series=None, slot_llm=None):
    if callable(slot_llm):
        return slot_llm(col, series)["C3"]
    return {"supply":0.0,"ret":0.0}

def c4_temporal_score(series, col=None, slot_llm=None):
    if callable(slot_llm):
        return slot_llm(col, series)["C4"]
    return {"variability":0.0}

def c5_group_consistency_score(df, col, slot_llm=None):
    if callable(slot_llm):
        return slot_llm(col, df[col])["C5"]
    return {"bundle":0.0}

# -----------------------
# C6：单位推测（强制使用 LLM 的 unit）
# -----------------------
def c6_unit_slot(col, series, slot_llm=None, unit_infer=None):
    """
    C6：单位信息
    优先级：
      1) LLM 给的 unit / unit_type / confidence
      2) header 中明显的单位（℃/kW/m3/h...）→ 轻微加conf
      3) 外部 unit_infer（数值分布推断）→ 在 LLM 不靠谱时兜底
    """
    name = str(col)
    unit = None
    unit_type = "unknown"
    conf = 0.0

    # 1) 从 LLM 读取 C6
    if callable(slot_llm):
        try:
            raw = slot_llm(col, series).get("C6", {})
            unit = raw.get("unit")
            unit_type = raw.get("unit_type") or "unknown"
            conf = float(raw.get("confidence") or 0.0)
        except Exception:
            pass

    # 2) 表头里有明显单位：小幅提高置信度（不改 unit_type）
    header_unit = _unit_from_header(name)
    if header_unit:
        # 如果 LLM 没给 unit，可以直接用 header 的 unit
        if not unit:
            unit = header_unit
        conf = min(1.0, conf + 0.1)

    # 3) 外部 unit_infer（比如你之前写的基于数值分布的单位推测）
    if callable(unit_infer):
        try:
            info = unit_infer(name, series) or {}
            u2 = info.get("unit")
            c2 = float(info.get("confidence") or 0.0)
            # 只有在 LLM 非常不自信的时候，才让外部推测覆盖
            if u2 and c2 > conf + 0.15:
                unit = u2
                conf = c2
                # unit_type 先不乱改，避免误伤；或者你可以在这里根据 u2 简单 map 一下
        except Exception:
            pass

    return {"unit": unit, "unit_type": unit_type, "confidence": conf}




# -----------------------
# C7：语义标签（LLM）
# -----------------------
def c7_llm_semantic(col, series=None, llm_func=None, slot_llm=None):
    if callable(slot_llm):
        return slot_llm(col, series)["C7"]
    return dict(tag=None, confidence=0.0)

# -----------------------
# C8：物理妥当性（LLM）
# -----------------------
def c8_physical_validity(df, col, label_guess=None, slot_llm=None):
    if callable(slot_llm):
        return slot_llm(col, df[col])["C8"]
    return dict(ok=0.5)


SLOT_WEIGHTS = {
    # favor semantics + physical validity
    "C1": 0.10, "C2": 0.18, "C3": 0.10, "C4": 0.00, "C5": 0.12, "C6": 0.12, "C7": 0.15, "C8": 0.15
}

# ROLE_KEYWORDS = {
#     "heat_source_supply_temp":  {"need_temp": True,  "need_supply": True,  "is_terminal": False},
#     "heat_source_return_temp":  {"need_temp": True,  "need_supply": False, "is_terminal": False},
#     "heat_source_flow":         {"need_flow": True,  "is_terminal": False},
#     "heat_source_power":        {"need_power": True, "is_terminal": False},
#     "heat_source_energy":       {"need_energy": True,"is_terminal": False},   # ★
#     "heat_source_capacity":     {"need_capacity": True, "is_terminal": False},
#
#     "terminal_supply_air_temp": {"need_temp": True,  "need_supply": True,  "is_terminal": True},
#     "terminal_return_air_temp": {"need_temp": True,  "need_supply": False, "is_terminal": True},
#     "terminal_air_volume":      {"need_flow": True,  "is_terminal": True},
#     "terminal_power":           {"need_power": True, "is_terminal": True},    # ★
#     "terminal_energy":          {"need_energy": True,"is_terminal": True},    # ★
#     "terminal_capacity":        {"need_capacity": True, "is_terminal": True},
# }



def score_slots_for_label(df, col, series, label, slot_llm=None, unit_infer=None):
    name = str(col)
    low = name.lower()
    upper = name.strip().upper()

    # ==== TIME / INDEX 硬过滤：所有角色得分都为 0，防止被误判为 power ====
    if _TIME_TOKEN.search(name) or upper in ("TIME", "INDEX"):
        detail = dict(C1=0.0, C2=0.0, C3=0.0, C4=0.0,
                      C5=0.0, C6=0.0, C7=0.0, C8=0.0, total=0.0)
        return 0.0, detail

    # ==== COP 列硬过滤：COP 是性能指标，不参与物理角色 ====
    if "cop" in low:
        detail = dict(C1=0.0, C2=0.0, C3=0.0, C4=0.0,
                      C5=0.0, C6=0.0, C7=0.0, C8=0.0, total=0.0)
        return 0.0, detail

    # ==== 强制：异常/状态/代码/予備 一律不参与任何物理角色 ====
    if _FAULT_TOKEN.search(name):
        detail = dict(C1=0.0, C2=0.0, C3=0.0, C4=0.0,
                      C5=0.0, C6=0.0, C7=0.0, C8=0.0, total=0.0)
        return 0.0, detail




    # 角色需求
    req = ROLE_KEYWORDS.get(label, {})
    need_temp    = req.get("need_temp", False)
    need_flow    = req.get("need_flow", False)
    need_power   = req.get("need_power", False)
    need_energy  = req.get("need_energy", False)
    need_cap     = req.get("need_capacity", False)

    if _OAT_TOKEN.search(name):
        if not need_temp:
            detail = dict(C1=0.0, C2=0.0, C3=0.0, C4=0.0,
                          C5=0.0, C6=0.0, C7=0.0, C8=0.0, total=0.0)
            return 0.0, detail

    # 表头关键字
    is_energy_header = bool(_KWH.search(low)) or ("日算値" in name) or ("日算" in name) or ("電力量" in name) or ("積算" in name)
    is_power_header  = ("電力" in name and "電力量" not in name and "日算値" not in name and "日算" not in name) \
                       or bool(_KW.search(low))
    is_temp_header   = bool(_TEMP_TOKEN.search(name))
    is_flow_header   = bool(_FLOW_TOKEN.search(name))
    # ==== 外气温度强制温度列（防止被 power 吸走） ====
    if _OAT_TOKEN.search(name):
        is_temp_header = True  # 强制给温度头信号
        is_power_header = False
        is_energy_header = False

    # ==== 日算/積算/電力量 强制能量列 ====
    if _DAILY_TOKEN.search(name):
        is_energy_header = True
        is_power_header = False

    if _FLOW_CMD_TOKEN.search(name):
        is_flow_header = False
        is_power_header = False
        is_energy_header = False
        unit_type = "control" if 'unit_type' in locals() else "control"

    # ==== 流量类强制 ====
    if _FLOW_HARD_TOKEN.search(name):
        is_flow_header = True
        is_power_header = False
        is_energy_header = False
        # 流量列不应该被 power/energy 头抢
        # （不强制 is_power_header=False，因为有些会写 kW? 但一般不）

    # ==== 流量类硬挡板：明显流量列禁止进入 power / energy 角色 ====
    if _FLOW_HARD_TOKEN.search(name):
        if need_power or need_energy:
            detail = dict(C1=0.0, C2=0.0, C3=0.0, C4=0.0,
                          C5=0.0, C6=0.0, C7=0.0, C8=0.0, total=0.0)
            return 0.0, detail





    # ==== 用 LLM 拿到 C1~C8 ====
    c1 = c1_component_score(col, series, slot_llm=slot_llm)
    c2 = c2_physical_quantity_score(col, series, slot_llm=slot_llm)
    c3 = c3_relation_score(col, series, slot_llm=slot_llm)
    c4 = c4_temporal_score(series, col, slot_llm=slot_llm)
    c5 = c5_group_consistency_score(df, col, slot_llm=slot_llm)
    c6 = c6_unit_slot(col, series, slot_llm=slot_llm, unit_infer=unit_infer)
    c7 = c7_llm_semantic(col, series, slot_llm=slot_llm)
    c8 = c8_physical_validity(df, col, slot_llm=slot_llm)

    # ==== C1: 组件类型 ====
    if req.get("is_terminal", False):
        s1 = float(c1.get("terminal", 0.0) or 0.0)
    else:
        s1 = max(
            float(c1.get("heat_source", 0.0) or 0.0),
            float(c1.get("fan", 0.0) or 0.0),
        )

    # ==== C6: 单位信息 ====
    unit_type = (c6.get("unit_type") or "unknown").lower()
    conf6 = float(c6.get("confidence", 0.0) or 0.0)
    s6 = conf6

    if conf6 < 0.6 and unit_type not in ("temp", "flow", "power", "energy", "pressure"):
        unit_type = "unknown"

    # 判断压力/控制列
    is_pressure_like = bool(_PRESSURE_TOKEN.search(name)) or (
        unit_type == "pressure" and not is_temp_header
    )
    is_control_like = (unit_type == "control") or bool(_INV_CTRL.search(name)) or bool(_FLOW_CMD_TOKEN.search(name))

    # ==== C2: 物理量类型 ====
    temp_score   = float(c2.get("temp", 0.0)   or 0.0)
    flow_score   = float(c2.get("flow", 0.0)   or 0.0)
    power_score  = float(c2.get("power", 0.0)  or 0.0)
    energy_score = float(c2.get("energy", 0.0) or 0.0)

    # ---------- C2 强否决逻辑（物理量严重不匹配直接 0 分） ----------
    c2_scores = {
        "temp":   temp_score,
        "flow":   flow_score,
        "power":  power_score,
        "energy": energy_score,
    }
    c2_main_type = max(c2_scores, key=lambda k: c2_scores[k])
    c2_main_val = c2_scores[c2_main_type]
    PHYS_STRONG = 0.7  # C2 非常确信阈值
    # ---------- 综合判定：是否“温度类列” ----------
    is_temp_like = is_temp_like_column(
        name,
        c2=c2,
        unit_type=unit_type,
    )

    if c2_main_val >= PHYS_STRONG:
        # 仅对 temp / flow / power / energy 角色做强否决，capacity 放过
        if need_temp   and c2_main_type != "temp":
            detail = dict(C1=0.0, C2=0.0, C3=0.0, C4=0.0,
                          C5=0.0, C6=0.0, C7=0.0, C8=0.0, total=0.0)
            return 0.0, detail
        if need_flow   and c2_main_type != "flow":
            detail = dict(C1=0.0, C2=0.0, C3=0.0, C4=0.0,
                          C5=0.0, C6=0.0, C7=0.0, C8=0.0, total=0.0)
            return 0.0, detail
        if need_power  and c2_main_type != "power":
            detail = dict(C1=0.0, C2=0.0, C3=0.0, C4=0.0,
                          C5=0.0, C6=0.0, C7=0.0, C8=0.0, total=0.0)
            return 0.0, detail
        if need_energy and c2_main_type != "energy":
            detail = dict(C1=0.0, C2=0.0, C3=0.0, C4=0.0,
                          C5=0.0, C6=0.0, C7=0.0, C8=0.0, total=0.0)
            return 0.0, detail

    # ---------- 正常 C2 打分逻辑 ----------
    s2 = 0.0
    if need_power:
        # ===== 关键修复：没有电力关键词/单位时，不允许 power 给满分（防止 power 吸铁石） =====
        has_power_evidence = is_power_header or (unit_type == "power") or bool(_KW.search(low)) or ("電力" in name)

        if not has_power_evidence:
            # 没有任何“电力”证据：power 只能给很低
            s2 = 0.0
        else:
            if is_power_header or unit_type == "power":
                s2 = 1.0
            elif power_score > 0.7 and energy_score < 0.3:
                s2 = 0.7


    elif need_energy:

        if is_energy_header or unit_type == "energy" or _DAILY_TOKEN.search(name):

            s2 = 1.0

        elif energy_score > 0.7 and power_score < 0.3:

            s2 = 0.7


    elif need_temp:

        if is_power_header:

            s2 = 0.0

        else:

            if is_temp_like or _OAT_TOKEN.search(name):

                s2 = 1.0

            elif temp_score > 0.7 and max(flow_score, power_score, energy_score) < 0.3:

                s2 = 0.7


    elif need_flow:

        if is_flow_header or unit_type == "flow" or _FLOW_HARD_TOKEN.search(name):
            s2 = 1.0

    elif need_cap:
        if _CAP_TOKEN.search(name):
            s2 = 1.0


    # ==== C3: supply / return ====
    s3 = _bool(
        (req.get("need_supply") and c3.get("supply", 0.0)) or
        ((req.get("need_temp") and not req.get("need_supply", False)) and c3.get("ret", 0.0)) or
        (not req.get("need_temp") and (c3.get("supply", 0.0) or c3.get("ret", 0.0)))
    )

    # ==== C4: 时序波动 ====
    s4 = float(c4.get("variability", 0.0) or 0.0)

    # ==== C5: bundle ====
    s5 = float(c5.get("bundle", 0.0) or 0.0)

    # ==== C6: unit_type 与角色需求的相性 ====
    if need_temp and unit_type not in ("temp", "unknown"):
        s6 = 0.0
    if need_power and unit_type not in ("power", "unknown"):
        s6 = 0.0
    if need_energy and unit_type not in ("energy", "unknown"):
        s6 = 0.0
    if need_flow and unit_type not in ("flow", "unknown"):
        s6 = 0.0

    # ==== 压力列：对所有 COP 相关角色一票否决 ====
    if is_pressure_like and (need_temp or need_power or need_flow or need_cap or need_energy):
        detail = dict(C1=0.0, C2=0.0, C3=0.0, C4=0.0,
                      C5=0.0, C6=0.0, C7=0.0, C8=0.0, total=0.0)
        return 0.0, detail

    # ==== 控制列：降低 COP 相关角色得分 ====
    if is_control_like and (need_temp or need_power or need_flow or need_cap or need_energy):
        s6 = 0.0
        if s2 > 0:
            s2 *= 0.2
    if unit_type == "control" and label != "other":
        detail = dict(C1=0.0, C2=0.0, C3=0.0, C4=0.0,
                      C5=0.0, C6=0.0, C7=0.0, C8=0.0, total=0.0)
        return 0.0, detail

    # ==== C7: tag 一致性 ====
    tag = c7.get("tag")
    s7 = 0.0
    if tag == label:
        s7 = float(c7.get("confidence", 0.5) or 0.5)
    elif tag in ("other", None):
        s7 = 0.1
    else:
        s7 = 0.0

    # ==== C8: 物理妥当性 ====
    s8 = float(c8.get("ok", 0.0) or 0.0)

    # ==== 总分 ====
    total = (
        SLOT_WEIGHTS["C1"] * s1 +
        SLOT_WEIGHTS["C2"] * s2 +
        SLOT_WEIGHTS["C3"] * s3 +
        SLOT_WEIGHTS["C4"] * s4 +
        SLOT_WEIGHTS["C5"] * s5 +
        SLOT_WEIGHTS["C6"] * s6 +
        SLOT_WEIGHTS["C7"] * s7 +
        SLOT_WEIGHTS["C8"] * s8
    )

    detail = dict(C1=s1, C2=s2, C3=s3, C4=s4, C5=s5, C6=s6, C7=s7, C8=s8, total=total)
    return total, detail



# 单独用于“列名 → 角色”的缓存，避免重复调用 Qwen
# 单独用于“列名 → 角色”的缓存，避免重复调用 Qwen
_NAME_LLM_CACHE = {}

def qwen_name_role(col_name: str, n_self_consistency: int = 3):
    """
    只根据「列名」让 Qwen 判定角色（带 self-consistency 多次投票）。
    返回: {"tag": <SLOT_LABELS 或 "other">, "confidence": 0.0~1.0}

    机制：
      1) 对同一列名调用 Qwen n_self_consistency 次；
      2) 用多数投票选 tag，取该 tag 的平均置信度；
      3) 若分歧较大，按分歧比例下调最终 confidence；
      4) 再套用你原来的 COP / energy / other 修正规则。
    """
    key = str(col_name)
    if key in _NAME_LLM_CACHE:
        return _NAME_LLM_CACHE[key]

    prompt = f"""
あなたは建築設備・HVACデータの専門家です。
次の列名が、以下の 13 個のどの役割に最も当てはまるかを判断してください。

列名: "{col_name}"

候補タグ（tag）は次の 13 個のいずれかです：
- "heat_source_supply_temp"
- "heat_source_return_temp"
- "heat_source_flow"
- "heat_source_power"
- "heat_source_energy"
- "heat_source_capacity"
- "terminal_supply_air_temp"
- "terminal_return_air_temp"
- "terminal_air_volume"
- "terminal_power"
- "terminal_energy"
- "terminal_capacity"
- "other"

注意：
- 列名に「COP」が含まれる場合、その列は性能指標であり物理量そのものではないため、必ず "other" に分類してください。
- 列名に「異常」「エラー」「コード」「予備」「保留」「dummy」などが含まれる場合も、原則 "other" としてください。
- 列名に「電力量」「積算」「日算値」「日算」「kWh」「Wh」「energy」が含まれる場合は、
  瞬時電力ではなく「積算電力量」を表す可能性が高いため、
  "heat_source_energy" または "terminal_energy" を優先的に検討してください。
- 列名に「消費電力（瞬時値）」「電力」「kW」が含まれ、
  「積算」「日算値」「電力量」などが含まれない場合は、瞬時電力（power）として扱ってください。

必ず JSON だけを返してください。形式は次の通りです：
{{
  "tag": "<上の候補タグのどれか1つ>",
  "confidence": 0.0 〜 1.0 の数値
}}

条件：
- tag は上記の 13 個のどれか「1つだけ」にしてください。
- confidence は、あなたの判断の自信度です（0.0〜1.0）。
- 追加の説明文やコメントは一切書かず、JSON のみを返してください。
"""

    # === self-consistency 多次投票 ===
    raw_results = []
    for i in range(max(1, n_self_consistency)):
        try:
            raw = qwen_chat_json(prompt)
            data_i = json.loads(raw)
            tag_i = data_i.get("tag")
            conf_i = data_i.get("confidence", 0.0)
            try:
                conf_i = float(conf_i)
            except Exception:
                conf_i = 0.0
            raw_results.append((tag_i, conf_i))
        except Exception as e:
            print(f"[QWEN-NAME ERROR] {col_name} (try {i+1}): {e}")

    if not raw_results:
        # 全部失败
        tag = "other"
        conf = 0.0
    else:
        # 统计投票 & 平均置信度
        from collections import Counter
        tags = [t for (t, c) in raw_results if t is not None]
        if not tags:
            tag = "other"
            conf = 0.0
        else:
            cnt = Counter(tags)
            majority_tag, max_count = cnt.most_common(1)[0]
            # 该 tag 的平均置信度
            majority_confs = [c for (t, c) in raw_results if t == majority_tag]
            mean_conf = sum(majority_confs) / len(majority_confs) if majority_confs else 0.0

            # 分歧比例：1 - (最多票数 / 总次数)
            disagree_ratio = 1.0 - (max_count / len(raw_results))
            # 分歧越大，惩罚越重（最多减到 50%）
            conf = mean_conf * (1.0 - 0.5 * disagree_ratio)
            tag = majority_tag

    # 剪裁 0~1
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    if not (0.0 <= conf <= 1.0):
        conf = 0.0

    name = str(col_name)
    low = name.lower()

    # ★ 强制：列名含 COP 一律归类为 other
    if "cop" in low:
        tag = "other"
        conf = max(conf, 0.9)

    # ★ 能量系关键字：积算 / 日算 / 電力量 / kWh / Wh / energy
    is_energy_header = (
        ("電力量" in name) or ("積算" in name) or ("日算値" in name) or ("日算" in name)
        or ("energy" in low) or re.search(r"\bkwh\b|\bwh\b", low)
    )
    if is_energy_header:
        # 粗略判断末端/热源
        terminal_like = any(k in name for k in ["AHU","FCU","室内機","fan","送風機","風機","blower","風","风","air"])
        energy_tag = "terminal_energy" if terminal_like else "heat_source_energy"

        # 如果 Qwen 没给 energy，就强行纠正为 energy
        if tag not in ("heat_source_energy", "terminal_energy"):
            tag = energy_tag
            conf = max(conf, 0.85)

    valid_tags = set(SLOT_LABELS) | {"other"}
    if tag not in valid_tags:
        tag = "other"

    result = {"tag": tag, "confidence": conf}
    _NAME_LLM_CACHE[key] = result
    return result


def _label_main_type(label: str) -> str:
    if not label or label == "other":
        return "other"
    l = label.lower()
    if "temp" in l:
        return "temp"
    if "flow" in l or "volume" in l:
        return "flow"
    if "energy" in l:
        return "energy"
    if "power" in l:
        return "power"
    if "capacity" in l:
        return "capacity"
    return "other"


def _apply_suspicious_rules(debug: dict, col_name: str):
    final_label = debug.get("final_label") or "other"
    final_type  = _label_main_type(final_label)

    if _FAULT_TOKEN.search(col_name) and final_label != "other":
        debug["suspicious_flag"] = True

    if _OAT_TOKEN.search(col_name) and final_type != "temp":
        debug["suspicious_flag"] = True

    if _DAILY_TOKEN.search(col_name) and final_type != "energy":
        debug["suspicious_flag"] = True

    if _FLOW_HARD_TOKEN.search(col_name) and final_type in ("power", "energy"):
        debug["suspicious_flag"] = True


def infer_role_by_slots(
    df,
    col,
    series,
    qwen_tag: str | None = None,
    qwen_conf: float | None = None,
    slot_llm=None,
    unit_infer=None,
    use_qwen_direct: bool = True,
    tau_high: float = 0.70,      # 插槽“高置信”阈值
    tau_low: float = 0.30,       # 插槽“低置信”阈值
    name_llm_high: float = 0.80, # 认为“高置信 Qwen”的阈值
    name_llm_low: float = 0.40,  # 目前主要用于日志和降级判断
):
    """
    综合 Qwen 列名直采 + C1–C8 插槽打分，推断一个字段的最终物理角色。
    """

    # === debug 初始化 ===
    debug = {
        "column": str(col),
        "qwen_tag": qwen_tag,
        "qwen_conf": qwen_conf,
        "qwen_type": None,
        "slot_best_label": None,
        "slot_best_score": None,
        "slot_type": None,
        "final_label": None,
        "final_score": 0.0,
        "decision": None,          # "qwen_high_conf" / "slot" / "other"
        "gate_status": None,      # "ACCEPT" / "ABSTAIN"
        "gate_reason": "",
        "eff_q": None,
        "eff_s": None,
        "eff_gap": None,
        "c8_best": None,
        "is_temp_like": None,
        "overconf_flag": False,
        "overconf_reason": "",
        "suspicious_flag": False,
        "explain_zh": "",
        "explain_ja": "",
    }
    def _finalize_and_return(label, score, per_label, debug):
        # 统一补全 suspicious_flag 规则
        _apply_suspicious_rules(debug, str(col))

        # 如果触发了 suspicious 规则，就别强行 False 覆盖
        if debug.get("suspicious_flag") is True:
            pass
        else:
            debug["suspicious_flag"] = bool(debug.get("suspicious_flag", False))

        return label, score, per_label, debug

    # === 0. 防御性处理 ===
    s = series
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        msg_zh = f"列「{col}」没有有效数值数据，无法进行物理判断，暂定为 other。"
        msg_ja = f"列「{col}」には有効な数値データがほとんどないため、物理的な判定ができず、一旦 other としました。"
        debug.update({
            "final_label": "other",
            "final_score": 0.0,
            "decision": "other",
            "suspicious_flag": True,
            "explain_zh": msg_zh,
            "explain_ja": msg_ja,
        })

        _apply_suspicious_rules(debug, str(col))  # ✅ 统一补全

        print(f"[DECISION] {col} 无有效数据，直接标记为 other")
        return _finalize_and_return("other", 0.0, {}, debug)
    try:
        sf = compute_shape_features(series)  # 注意：用原 series，不要用 dropna 后的 s 也行
        if sf.get("is_all_zero") or (sf.get("zero_ratio") is not None and sf["zero_ratio"] >= 0.98):
            name = str(col)
            low = name.lower()

            has_strong_unit_or_kw = (
                    ("kw" in low) or ("kwh" in low) or ("wh" in low) or ("℃" in name) or ("°c" in low)
            )
            has_strong_kw = bool(
                _KWH.search(low) or _KW.search(low) or _TEMP_TOKEN.search(name) or _FLOW_TOKEN.search(name)
            )

            if not (has_strong_unit_or_kw or has_strong_kw):
                msg_zh = f"列「{col}」几乎全为0（zero_ratio={sf['zero_ratio']:.2f}），更像停机/无效数据，暂定为 other（需人工确认）。"
                msg_ja = f"列「{col}」はほぼ0（zero_ratio={sf['zero_ratio']:.2f}）で、停止/無効データの可能性が高いため、一旦 other（要確認）としました。"
                debug.update({
                    "final_label": "other",
                    "final_score": 0.0,
                    "decision": "other",
                    "suspicious_flag": True,
                    "explain_zh": msg_zh,
                    "explain_ja": msg_ja,
                })

                _apply_suspicious_rules(debug, str(col))
                print(f"[DECISION] {col} 几乎全0 → 直接 other（suspicious）")
                return _finalize_and_return("other", 0.0, {}, debug)
    except Exception:
        pass


    if unit_infer is None:
        def unit_infer(_name, _series):
            return {"unit": None, "confidence": 0.0}

    # === 新增：全局 C2 物理类型（一次性从 slot_llm 取，用缓存不会重复算） ===
    c2_main_type = "other"
    c2_main_val = 0.0
    if callable(slot_llm):
        try:
            slot_all = slot_llm(col, series) or {}
            c2 = slot_all.get("C2", {}) or {}
            temp_score   = float(c2.get("temp", 0.0)   or 0.0)
            flow_score   = float(c2.get("flow", 0.0)   or 0.0)
            power_score  = float(c2.get("power", 0.0)  or 0.0)
            energy_score = float(c2.get("energy", 0.0) or 0.0)
            c2_scores = {
                "temp":   temp_score,
                "flow":   flow_score,
                "power":  power_score,
                "energy": energy_score,
            }
            c2_main_type = max(c2_scores, key=lambda k: c2_scores[k])
            c2_main_val  = c2_scores[c2_main_type]
        except Exception:
            pass

    # === 1. 先跑 C1–C8 插槽打分，找出 slot_best_label ===
    best_label = "other"
    best_score = 0.0
    per_label: dict[str, dict] = {}

    for lab in SLOT_LABELS:
        sc, det = score_slots_for_label(
            df, col, series, lab,
            slot_llm=slot_llm,
            unit_infer=unit_infer,
        )
        per_label[lab] = det
        if sc > best_score:
            best_score, best_label = sc, lab

    slot_best_label = best_label
    slot_best_score = best_score
    print(f"[SLOT] {col} → {slot_best_label}:{slot_best_score:.2f}")

    # === 2. 角色 -> 主物理类型 映射函数 ===
    def _role_main_type(role: str) -> str:
        if not role or str(role).lower() == "other":
            return "other"

        meta = ROLE_KEYWORDS.get(role) or {}

        def _from_meta(m):
            if not isinstance(m, dict):
                return None
            if m.get("need_temp"):     return "temp"
            if m.get("need_flow"):     return "flow"
            if m.get("need_power"):    return "power"
            if m.get("need_energy"):   return "energy"
            if m.get("need_capacity"): return "capacity"
            return None

        t = _from_meta(meta)
        if t:
            return t

        r = (role or "").lower()
        if "temp" in r:                      return "temp"
        if "flow" in r or "volume" in r:     return "flow"
        if "power" in r:                     return "power"
        if "energy" in r:                    return "energy"
        if "capacity" in r:                  return "capacity"
        return "other"

    slot_type = _role_main_type(slot_best_label)
    slot_dev_type = _role_device_type(slot_best_label)  # ★ 新增：slot 设备类型

    debug["slot_best_label"] = slot_best_label
    debug["slot_best_score"] = slot_best_score
    debug["slot_type"] = slot_type
    debug["slot_device_type"] = slot_dev_type  # ★ 记到 debug 里

    # === 3. 没有 / 不用 Qwen → 完全靠插槽 ===
    if (not use_qwen_direct) or (qwen_tag is None) or (qwen_conf is None):
        if slot_best_score < tau_low:
            msg_zh = f"列「{col}」没有 Qwen 直采结果，且插槽打分也偏低（{slot_best_score:.2f}），暂定为 other，建议人工确认。"
            msg_ja = f"列「{col}」は Qwen 列名判定がなく、スロットスコアも低いため（{slot_best_score:.2f}）、一旦 other とし、目視確認を推奨します。"
            debug.update({
                "final_label": "other",
                "final_score": slot_best_score,
                "decision": "other",
                "suspicious_flag": True,
                "explain_zh": msg_zh,
                "explain_ja": msg_ja,
            })
            print(f"[DECISION] {col} 无 Qwen / 未使用直采，且 slot 置信度过低({slot_best_score:.2f}) → other")
            return "other", slot_best_score, per_label, debug

        msg_zh = f"列「{col}」没有 Qwen 列名先验，最终角色由 C1–C8 插槽综合判断为 {slot_best_label}（score={slot_best_score:.2f}）。"
        msg_ja = f"列「{col}」は Qwen 列名の事前情報がないため、C1〜C8 スロットの総合評価により {slot_best_label}（score={slot_best_score:.2f}）と判定しました。"
        debug.update({
            "final_label": slot_best_label,
            "final_score": slot_best_score,
            "decision": "slot",
            "suspicious_flag": False,
            "explain_zh": msg_zh,
            "explain_ja": msg_ja,
        })
        print(f"[DECISION] {col} 无 Qwen / 未使用直采 → 使用 slot 结果: {slot_best_label} ({slot_best_score:.2f})")
        return _finalize_and_return(slot_best_label, slot_best_score, per_label, debug)


    # === 4. 有 Qwen 的情况：先做类型+置信度检查 ===
    qwen_tag = qwen_tag.strip() if isinstance(qwen_tag, str) else qwen_tag
    qwen_type = _role_main_type(qwen_tag)
    qwen_dev_type = _role_device_type(qwen_tag)          # ★ 新增：Qwen 设备类型
    q_conf_adj = float(qwen_conf)

    debug["qwen_tag"] = qwen_tag
    debug["qwen_conf"] = qwen_conf
    debug["qwen_type"] = qwen_type
    debug["qwen_device_type"] = qwen_dev_type

    print(f"[QWEN-NAME] {col} → tag={qwen_tag}, conf={qwen_conf:.2f}, type={qwen_type}")

    # 4.1 初始判断：高置信 or 非高置信
    if q_conf_adj < name_llm_high:
        print(
            f"[QWEN-CHECK] {col} Qwen conf={q_conf_adj:.2f} < 高置信阈值({name_llm_high})，"
            f"不作为强先验，仅供参考 → 进入插槽决策阶段"
        )
        # 后面统一按“插槽主导”再综合
    else:
        # === 4.2 Qwen 高置信 → 用 AI 信号加权比较 Qwen vs Slot，而不是直接说 Qwen 过度自信 ===


        # 0) 额外算一个“这个列是不是温度列”的信号（名字 + C2 + unit_type）
        is_temp_like = False
        try:
            c2_for_temp = None
            unit_type_for_temp = None

            if callable(slot_llm):
                slot_all2 = slot_llm(col, series) or {}
                c2_for_temp = slot_all2.get("C2", {}) or {}
                c6_for_temp = slot_all2.get("C6", {}) or {}
                unit_type_for_temp = (c6_for_temp.get("unit_type") or "unknown")

            is_temp_like = is_temp_like_column(str(col), c2=c2_for_temp, unit_type=unit_type_for_temp)
        except Exception:
            is_temp_like = False


        # 1) Qwen 侧证据：列名置信度 + 对该 tag 的插槽得分
        qwen_slot_score = per_label.get(qwen_tag, {}).get("total", 0.0) if qwen_tag in per_label else 0.0
        eff_q = 0.6 * q_conf_adj + 0.4 * qwen_slot_score

        # 2) Slot 侧证据：slot_best_score
        eff_s = slot_best_score

        # 3) 看 C2 物理量类型站哪一边（完全来自 slot_llm，不靠手写关键词）
        if c2_main_type == qwen_type and c2_main_val > 0.5:
            eff_q += 0.15 * c2_main_val  # C2 支持 Qwen 这边

        # 对 slot 的 C2 加成：如果当前列“看起来像温度”，且 Qwen 是 temp，而 slot 是 power/energy，
        # 就不要给 slot 这份 C2 bonus（否则会放大“把温度当电力”的错误）
        if c2_main_type == slot_type and c2_main_val > 0.5:
            if not (is_temp_like and qwen_type == "temp" and slot_type in ("power", "energy")):
                eff_s += 0.15 * c2_main_val  # C2 支持 slot 这边

        # 3.5) header 级别的先验：名字/C2 像温度 → 偏向温度解读
        if is_temp_like and qwen_type == "temp":
            eff_q += 0.2  # 给 Qwen 额外加一个“这列是温度”的加成
        if is_temp_like and slot_type in ("power", "energy"):
            eff_s = max(0.0, eff_s - 0.15)  # 同时温和地削一点 slot 的权重

        print(
            f"[QWEN-VS-SLOT] {col} eff_q={eff_q:.2f} (q_tag={qwen_tag}, type={qwen_type}) | "
            f"eff_s={eff_s:.2f} (slot={slot_best_label}, type={slot_type}), "
            f"C2={c2_main_type}:{c2_main_val:.2f}, temp_like={is_temp_like}"
        )



        # --- Gate bookkeeping: explicit ACCEPT/ABSTAIN ---
        c8_best = float(per_label.get(slot_best_label, {}).get("C8", 0.0) or 0.0)
        eff_gap = abs(eff_q - eff_s)
        debug.update({
            "eff_q": eff_q,
            "eff_s": eff_s,
            "eff_gap": eff_gap,
            "c8_best": c8_best,
            "is_temp_like": is_temp_like,
        })

        gate_min_c8 = 0.35
        gate_min_gap = 0.08
        gate_strong = 0.75  # evidence strong enough to proceed even if gap is small

        gate_status = "ACCEPT"
        gate_reason = ""

        if c8_best < gate_min_c8:
            gate_status = "ABSTAIN"
            gate_reason = f"low_physical_validity(C8={c8_best:.2f}<{gate_min_c8})"
        elif eff_gap < gate_min_gap and max(eff_q, eff_s) < gate_strong:
            gate_status = "ABSTAIN"
            gate_reason = f"ambiguous_evidence(eff_gap={eff_gap:.2f}<{gate_min_gap})"

        debug["gate_status"] = gate_status
        debug["gate_reason"] = gate_reason

        # If we abstain (and Qwen did not explicitly say 'other'), force 'other' and explain why.
        if gate_status == "ABSTAIN" and qwen_tag != "other":

            name = str(col)

            # ✅ 判断是否是“强语义列”
            strong_name_hint = any(k in name for k in [
                "流量", "風量", "温度", "電力", "能力", "COP", "圧力"
            ])

            if strong_name_hint:
                # ❗保留原标签，不强制变 other
                debug.update({
                    "decision": "soft_abstain",
                    "suspicious_flag": True,
                    "explain_zh": f"{col} 语义明确但物理验证弱 → 保留标签",
                    "explain_ja": f"{col} は意味が明確だが物理検証が弱いため保持",
                })

                print(f"[SOFT GATE] {col} 保留标签（低置信）")
                soft_label = qwen_tag if qwen_tag and qwen_tag != "other" else slot_best_label
                soft_score = max(q_conf_adj, slot_best_score or 0.0)

                debug.update({
                    "final_label": soft_label,
                    "final_score": soft_score,
                    "decision": "soft_abstain",
                    "suspicious_flag": True,
                    "explain_zh": f"{col} 语义明确但物理验证弱 → 保留标签 {soft_label}",
                    "explain_ja": f"{col} は意味が明確だが物理検証が弱いため {soft_label} として保持",
                })

                print(f"[SOFT GATE] {col} 保留标签: {soft_label}（低置信）")
                return _finalize_and_return(soft_label, soft_score, per_label, debug)

            # ❗原来的逻辑（只对垃圾列生效）
            debug.update({
                "final_label": "other",
                "decision": "hard_abstain",
                "suspicious_flag": True,
            })

            print(f"[HARD GATE] {col} → other")
            return _finalize_and_return("other", max(slot_best_score, q_conf_adj), per_label, debug)

        # ===== PATCH: 高置信 other 的否决权（防止体系外变量被硬塞） =====
        # 条件：
        # 1) Qwen 高置信
        # 2) Qwen 判为 other（明确“不属于当前角色体系”）
        # 3) slot 并没有明显更强
        if (
            qwen_tag == "other"
            and q_conf_adj >= name_llm_high
            and eff_s < eff_q + 0.15   # slot 没有明显优势
        ):
            msg_zh = (
                f"列「{col}」被 Qwen 高置信判定为 other（不属于当前物理角色体系），"
                f"插槽结果优势不足（eff_s={eff_s:.2f}, eff_q={eff_q:.2f}），"
                f"为避免误归类，最终保留为 other。"
            )
            msg_ja = (
                f"列「{col}」は Qwen により高信頼度で other（本ロール体系外）と判定され、"
                f"スロット側の優位性が不十分なため（eff_s={eff_s:.2f}, eff_q={eff_q:.2f}）、"
                f"誤分類防止のため other を維持します。"
            )
            debug.update({
                "final_label": "other",
                "final_score": q_conf_adj,
                "decision": "qwen_other_veto",
                "suspicious_flag": False,
                "explain_zh": msg_zh,
                "explain_ja": msg_ja,
            })
            print(f"[DECISION] {col} Qwen 高置信 other → 否决 slot，保留 other")
            return _finalize_and_return("other", q_conf_adj, per_label, debug)
        # ===== PATCH END =====




        # 4) 若 Qwen 侧综合证据 ≥ slot 侧，则直接采用 Qwen
        if eff_q >= eff_s:
            if use_qwen_direct:
                if qwen_tag in per_label:
                    best_score = per_label[qwen_tag].get("total", q_conf_adj)
                else:
                    best_score = q_conf_adj

                msg_zh = (
                    f"列「{col}」中，Qwen 列名判定与 C1–C8/物理类型/列名综合后支持度更高 "
                    f"(eff_q={eff_q:.2f} ≥ eff_s={eff_s:.2f})，因此采用 Qwen 结果 {qwen_tag}。"
                )
                msg_ja = (
                    f"列「{col}」では、Qwen の列名判定と C1〜C8/物理タイプ/列名の総合評価の支持度が "
                    f"スロット側より高いため (eff_q={eff_q:.2f} ≥ eff_s={eff_s:.2f})、"
                    f"最終ラベルとして {qwen_tag} を採用しました。"
                )
                debug.update({
                    "final_label": qwen_tag,
                    "final_score": best_score,
                    "decision": "qwen_weighted_win",
                    "suspicious_flag": False,
                    "explain_zh": msg_zh,
                    "explain_ja": msg_ja,
                })
                print(
                    f"[DECISION] {col} Qwen 综合证据更强 → 采用 Qwen: {qwen_tag} "
                    f"(conf={qwen_conf:.2f}, eff_q={eff_q:.2f})"
                )
                return _finalize_and_return(qwen_tag, best_score, per_label, debug)


        # 5) 只有在 slot 证据明显更强时，才把 Qwen 当作“过度自信”降级
        margin = 0.15  # slot 至少要比 Qwen 多 0.15 才算真正赢
        if eff_s >= eff_q + margin and slot_best_score >= tau_high:
            overconf_reason = (
                f"slot 综合证据明显强于 Qwen (eff_s={eff_s:.2f} ≥ eff_q={eff_q:.2f}+{margin})，"
                f"且 slot_best_score={slot_best_score:.2f} ≥ tau_high={tau_high}"
            )
            print(f"[QWEN-OVERCNF] {col} {overconf_reason} → 将 Qwen 降级为低置信")
            debug["overconf_flag"] = True
            debug["overconf_reason"] = overconf_reason
            q_conf_adj = max(0.0, name_llm_low - 1e-3)
        else:
            print(
                f"[QWEN-MED] {col} Qwen/slot 证据接近或互相牵制 (eff_q={eff_q:.2f}, eff_s={eff_s:.2f})，"
                f"不视为 Qwen 过度自信，后续由插槽结果主导。"
            )


    # === 5. 走到这里：Qwen 非高置信 / 已被降级 → 插槽主导 ===
    # --- Gate bookkeeping for slot-dominant path ---
    c8_best = float(per_label.get(slot_best_label, {}).get("C8", 0.0) or 0.0)
    debug["c8_best"] = c8_best
    gate_min_c8 = 0.35
    if c8_best < gate_min_c8:
        msg_zh = (
            f"列「{col}」的物理妥当性不足（C8={c8_best:.2f}<{gate_min_c8}），"
            "触发 gate 机制 → 暂定为 other（建议人工核对）。"
        )
        msg_ja = (
            f"列「{col}」は物理的妥当性が低く（C8={c8_best:.2f}<{gate_min_c8}）、"
            "gate を発動して一旦 other（要確認）としました。"
        )
        debug.update({
            "final_label": "other",
            "final_score": max(slot_best_score, q_conf_adj),
            "decision": "gate_abstain",
            "gate_status": "ABSTAIN",
            "gate_reason": f"low_physical_validity(C8={c8_best:.2f}<{gate_min_c8})",
            "suspicious_flag": True,
            "explain_zh": msg_zh,
            "explain_ja": msg_ja,
        })
        print(f"[GATE] {col} ABSTAIN (slot-dominant) → other | C8={c8_best:.2f}")
        return "other", max(slot_best_score, q_conf_adj), per_label, debug

    if slot_best_score < tau_low:
        # 插槽也没把握 → 标记为 suspicious
        msg_zh = (
            f"列「{col}」的 Qwen 置信度不足或已被判定为过度自信，"
            f"同时插槽得分也较低（{slot_best_score:.2f}），无法可靠分类，暂定为 other，建议人工核对。"
        )
        msg_ja = (
            f"列「{col}」は Qwen の信頼度が低い（または過信と判定）うえに、"
            f"スロットスコアも低いため（{slot_best_score:.2f}）、一旦 other とし、目視確認を推奨します。"
        )
        debug.update({
            "final_label": "other",
            "final_score": max(slot_best_score, q_conf_adj),
            "decision": "other",
            "suspicious_flag": True,
            "explain_zh": msg_zh,
            "explain_ja": msg_ja,
        })
        print(
            f"[DECISION] {col} Qwen 非高置信/已降级(q={q_conf_adj:.2f}) 且 slot_best_score 过低({slot_best_score:.2f}) "
            f"→ 标记为 other"
        )
        return "other", max(slot_best_score, q_conf_adj), per_label, debug

    # 插槽有一定把握 → 完全采用插槽结果
    msg_zh = (
        f"列「{col}」的 Qwen 结果置信度不足/已降级，"
        f"插槽综合得分较高（{slot_best_score:.2f}），最终采用 C1–C8 插槽判断结果 {slot_best_label}。"
    )
    msg_ja = (
        f"列「{col}」は Qwen の信頼度が十分でない（または過信と判定された）ため、"
        f"C1〜C8 スロットの総合スコア（{slot_best_score:.2f}）を優先し、最終ラベルを {slot_best_label} としました。"
    )
    debug.update({
        "final_label": slot_best_label,
        "final_score": slot_best_score,
        "decision": "slot",
        "suspicious_flag": debug.get("overconf_flag", False),  # 若发生过度自信则标记可疑
        "explain_zh": msg_zh,
        "explain_ja": msg_ja,
    })

    print(
        f"[DECISION] {col} Qwen 非高置信/已降级(q={q_conf_adj:.2f})，"
        f"slot_best_score={slot_best_score:.2f} >= tau_low({tau_low}) → 使用 slot: {slot_best_label}"
    )
    return slot_best_label, slot_best_score, per_label, debug










def ai_judge_physical_role(col, client, verbose=True):
    """
    AI 判定列角色；client 可为 None。
    - client 为 None 时直接返回 "other"（由上层兜底规则决定最终角色）
    - 返回值强约束在 valid_tags 之内
    """
    valid_tags = [
        "heat_source_supply_temp", "heat_source_return_temp", "heat_source_flow",
        "heat_source_power", "heat_source_energy", "heat_source_capacity",
        "terminal_supply_air_temp", "terminal_return_air_temp", "terminal_air_volume",
        "terminal_power", "terminal_energy", "terminal_capacity", "other"
    ]
    if client is None:
        if verbose:
            print(f"[AI判定跳过] {col} (client=None)")
        return "other"

    prompt = (
        "你是建筑暖通能效专家。"
        "请将下列列名严格分类，只能属于下列13类之一（只返回英文标签，不要解释）：\n"
        "【热源相关】\n"
        "1. heat_source_supply_temp\n"
        "2. heat_source_return_temp\n"
        "3. heat_source_flow\n"
        "4. heat_source_power\n"
        "5. heat_source_energy\n"
        "6. heat_source_capacity\n"
        "【末端相关】\n"
        "7. terminal_supply_air_temp\n"
        "8. terminal_return_air_temp\n"
        "9. terminal_air_volume\n"
        "10. terminal_power\n"
        "11. terminal_energy\n"
        "12. terminal_capacity\n"
        "【其它】\n"
        "13. other\n"
        f"列名：{col}"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是暖通数据专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.01,
            max_tokens=16,
        )
        ans = (resp.choices[0].message.content or "").strip().lower()
    except Exception as e:
        if verbose:
            print(f"[AI判定异常] {col}: {e}")
        ans = "other"

    if ans not in valid_tags:
        if verbose:
            print(f"[AI判定越界] {col} => {ans} (fallback=other)")
        ans = "other"
    else:
        if verbose:
            print(f"[AI判定] {col} => {ans}")
    return ans


def batch_physical_role_review(
    df,
    client=None,
    role_db_path="hvac_physical_role_db.json",
    export_slot_csv: bool = False,
    slot_csv_path: str | None = None,
    unit_infer=None,
):
    """
    返回:
      review_dict: 仅包含与 COP/负荷率直接相关的字段 -> 角色
      ai_roles:     全部列 -> 角色（AI 或本地兜底）
      slot_details: 每列的插槽 & debug 明细，结构:
                    {
                      col_name: {
                        "per_label": {...},
                        "debug": {...}
                      },
                      ...
                    }

    新增：
    - 当 export_slot_csv=True 时，额外导出每列的调试信息 CSV；
    - 自动输出“怀疑列列表”（suspicious_flag=True）；
    - slot_details 可用于后续绘图 / GUI 调试。
    """
    try:
        db = json.load(open(role_db_path, "r", encoding="utf-8")) if os.path.exists(role_db_path) else {}
    except Exception:
        db = {}

    def _fallback_role(col: str) -> str:
        # ……这里保留你原来的兜底逻辑，不改……
        name = str(col)
        low = name.lower()

        if any(k in name for k in
               ["status", "alarm", "cmd", "command", "on/off", "mode", "異常", "指令", "コード", "sp", "設定"]) \
                or _TIME_TOKEN.search(name):
            return "other"

        if any(k in name for k in ["圧力"]) or _PRESSURE_TOKEN.search(low):
            return "other"

        terminal_like = any(
            k in name for k in ["AHU", "FCU", "室内機", "fan", "送風機", "風機", "blower", "風", "风", "air"])

        is_energy_header = (
                ("電力量" in name) or ("積算" in name) or ("日算値" in name) or ("日算" in name)
                or ("energy" in low) or re.search(r"\bkwh\b|\bwh\b", low)
        )
        if is_energy_header:
            role = "terminal_energy" if terminal_like else "heat_source_energy"
            print(f"[ENERGY-DETECT] {col} → {role} (fallback)")
            return role

        is_power_header = (
                ("電力" in name and "電力量" not in name and "日算値" not in name and "日算" not in name)
                or re.search(r"\bkw\b", low)
        )
        if is_power_header:
            role = "terminal_power" if terminal_like else "heat_source_power"
            return role

        is_temp = any(k in name for k in ["温度", "℃", "°c"]) or ("temp" in low) or ("temperature" in low)
        if is_temp:
            if any(k in name for k in ["往", "出", "吹", "送", "供", "出口"]) or ("supply" in low) or (
                    "flow out" in low):
                if any(k in name for k in ["風", "风"]) or ("air" in low):
                    return "terminal_supply_air_temp"
                return "heat_source_supply_temp"
            if any(k in name for k in ["還", "回", "吸", "入口"]) or ("return" in low) or ("inlet" in low) or (
                    "flow in" in low):
                if any(k in name for k in ["風", "风"]) or ("air" in low):
                    return "terminal_return_air_temp"
                return "heat_source_return_temp"
            return "temperature_other"

        if any(k in name for k in ["流量", "m³/h", "m3/h", "L/min", "l/min", "l/s", "L/s"]) or (
                "flow" in low) or ("air volume" in low):
            if any(k in name for k in ["風", "风"]) or ("air" in low):
                return "terminal_air_volume"
            return "heat_source_flow"

        if any(k in name for k in ["能力", "容量"]) or ("capacity" in low):
            if any(k in name for k in ["風", "风"]) or ("air" in low):
                return "terminal_capacity"
            return "heat_source_capacity"

        return "other"

    def _ai_try(col: str):
        try:
            if client is None:
                return None
            return ai_judge_physical_role(col, client, verbose=False)
        except Exception:
            return None

    csv_rows = [] if export_slot_csv else None

    ai_roles = {}
    slot_details = {}
    neighbor_cols = list(df.columns)

    def slot_llm(col, series):
        return llm_score_all_slots(col, series, neighbor_cols=neighbor_cols)

    # === 主循环：每一列做一次 Qwen + C1~C8 推断 ===
    for col in df.columns:
        name_info = qwen_name_role(col)  # {"tag": ..., "confidence": ...}
        label, score, per_label, dbg = infer_role_by_slots(
            df=df,
            col=col,
            series=df[col],
            qwen_tag=name_info["tag"],
            qwen_conf=name_info["confidence"],
            slot_llm=slot_llm,
            unit_infer=unit_infer,
        )

        # 若 infer_role_by_slots 返回 other 且插槽很低，可以考虑兜底关键字/AI
        # 但如果 gate 明确 ABSTAIN（不允许推理），则不要用关键字兜底强行贴标签。
        if label == "other" and score < 0.3 and dbg.get("gate_status") != "ABSTAIN":
            fb = _fallback_role(col)
            if fb != "other":
                print(f"[FALLBACK] {col}: slot/LLM 给出 other, 关键字兜底为 {fb}")
                label = fb

        ai_roles[col] = label
        slot_details[col] = {
            "per_label": per_label,
            "debug": dbg,
        }

        if export_slot_csv and csv_rows is not None:
            row = build_slot_debug_row(col, label, score, per_label, dbg)
            csv_rows.append(row)

    whitelist = {
        "heat_source_supply_temp", "heat_source_return_temp", "heat_source_flow",
        "heat_source_power", "heat_source_capacity",
        "terminal_supply_air_temp", "terminal_return_air_temp", "terminal_air_volume",
        "terminal_power", "terminal_capacity",
        "heat_source_energy", "terminal_energy",
    }
    review_dict = {c: r for c, r in ai_roles.items() if r in whitelist}

    try:
        db.update(ai_roles)
        json.dump(db, open(role_db_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    except Exception:
        pass

    # 导出 CSV
    if export_slot_csv and csv_rows is not None:
        base_dir = r"D:\dynamoFile\PythonScript\LLM\output\slot"
        os.makedirs(base_dir, exist_ok=True)
        out_path = slot_csv_path or os.path.join(base_dir, "slot_scores_latest.csv")

        try:
            pd.DataFrame(csv_rows).to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"[LOG] 插槽调试信息已导出到: {out_path}")
        except Exception as e:
            print(f"[WARN] 写入 CSV 失败: {e}")

    # === 怀疑列列表（suspicious_flag=True） ===
    suspicious_cols = [
        col for col, info in slot_details.items()
        if info.get("debug", {}).get("suspicious_flag")
    ]
    if suspicious_cols:
        print("\n[SUSPECT] 需要人工确认的列:")
        for c in suspicious_cols:
            dbg = slot_details[c]["debug"]
            print(f"  - {c}: {dbg.get('final_label')} | {dbg.get('overconf_reason') or dbg.get('explain_zh')}")

    cop_role_whitelist = {
        "heat_source_supply_temp","heat_source_return_temp","heat_source_flow",
        "heat_source_power","heat_source_capacity",
        "terminal_supply_air_temp","terminal_return_air_temp","terminal_air_volume",
        "terminal_power","terminal_capacity",
    }
    energy_roles = {"heat_source_energy","terminal_energy"}

    cop_cols = [c for c, r in ai_roles.items() if r in cop_role_whitelist]
    energy_cols = [c for c, r in ai_roles.items() if r in energy_roles]

    print(f"\n[COP-COLS] 将参与 COP/负荷率计算的列: {cop_cols}")
    print(f"[ENERGY-COLS] 被识别为 積算/日算 電力量 的列(不直接参与 COP): {energy_cols}")
    print("\n=== 批量自动分类完成（Qwen高置信 + 插槽物理检查 + debug 输出）===")
    return review_dict, ai_roles, slot_details



# analysis_core.py 增加
def ai_pick_cop_combinations_for_group(col_list, group_name, lang='JP', client=None):
    prompt = (
        f"你是暖通空调（HVAC）数据专家。\n"
        f"以下是某台设备（组名: {group_name}）的数据列名列表：\n"
        + "\n".join(col_list) +
        "\n请你基于列名，自动生成【热源COP】和【末端COP】各自可能的物理量字段组合，"
        "每组字段要能支持COP/负荷率计算。每个组合请附带你对该组合的置信度（0~1），"
        "并用如下JSON格式输出：\n"
        "{\n"
        "  \"heat_source_cop\": [\n"
        "    {\n"
        "      \"fields\": {\"supply_temp\": \"xxx\", \"return_temp\": \"yyy\", \"flow\": \"zzz\", \"power\": \"aaa\"},\n"
        "      \"confidence\": 0.92\n"
        "    }, ...\n"
        "  ],\n"
        "  \"terminal_cop\": [\n"
        "    {\n"
        "      \"fields\": {\"supply_air_temp\": \"xxx\", \"return_air_temp\": \"yyy\", \"air_volume\": \"zzz\", \"terminal_power\": \"aaa\"},\n"
        "      \"confidence\": 0.86\n"
        "    }, ...\n"
        "  ]\n"
        "}\n"
        "注意：如果某些字段找不到就用null。尽量给出复数组不同组合与置信度。只输出JSON，不要加解释。"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是HVAC物理量字段组合推荐助手，输出严格JSON。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=512,
    )
    result = response.choices[0].message.content
    return robust_json_parse(result)


def robust_json_parse(gpt_output):
    """
    强健 JSON 解析器：
    - 兼容 ```json ...``` 围栏
    - 抽取最外层 {...}
    - 单引号->双引号、尾逗号修正
    - 修复 "key":"val" 中 val 里未转义的英文引号
    """
    if not gpt_output or not str(gpt_output).strip():
        print("[ERROR] AI输出为空")
        return {}

    content = str(gpt_output).strip()

    # 去掉代码块
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content, re.IGNORECASE)
    if m:
        content = m.group(1)

    # 抽取最外层 {...}
    if "{" in content and "}" in content:
        content = content[content.find("{"): content.rfind("}") + 1]

    # 单引号 -> 双引号；尾逗号 -> 去除
    content = content.replace("'", '"')
    content = re.sub(r",\s*([}\]])", r"\1", content)

    # 修复 value 内未转义的 "
    # 只处理 "key":"value" 的 value 部分，把 value 内部裸引号替换为 【】 以避免 json 失败
    def _fix_value_quotes(match):
        value = match.group(2)
        fixed = re.sub(r'(?<!\\)"', '”', value)  # 或替换为全角/双引号，避免破坏 JSON 结构
        return f'{match.group(1)}"{fixed}"'

    content = re.sub(r'(".*?":\s*")([^"]*(".*?")+[^"]*)"', _fix_value_quotes, content)

    try:
        return json.loads(content)
    except Exception as e:
        print(f"[JSON解析失败] {e}\n内容:\n{content[:1000]}")
        return {}




def get_slot_debug_info_from_cache(slot_details):
    """
    直接用 batch_physical_role_review 返回的 slot_details，
    不再重新调用 infer_role_by_slots。
    """
    return slot_details

def build_slot_debug_row(col, final_label, final_score, per_label, debug):
    """
    把某一列的最终结果 + 插槽明细 + Qwen/Slot 对比信息
    打平成一行 dict，用于写入 CSV。
    """
    row = {
        "column": str(col),
        "final_label": final_label,
        "final_score": final_score,
        "decision": debug.get("decision"),
        "suspicious_flag": debug.get("suspicious_flag"),
        "qwen_tag": debug.get("qwen_tag"),
        "qwen_conf": debug.get("qwen_conf"),
        "qwen_type": debug.get("qwen_type"),
        "slot_best_label": debug.get("slot_best_label"),
        "slot_best_score": debug.get("slot_best_score"),
        "slot_type": debug.get("slot_type"),
        "overconf_flag": debug.get("overconf_flag"),
        "overconf_reason": debug.get("overconf_reason"),
        "explain_zh": debug.get("explain_zh"),
        "explain_ja": debug.get("explain_ja"),
        "gate_status": debug.get("gate_status"),
        "gate_reason": debug.get("gate_reason"),
        "eff_q": debug.get("eff_q"),
        "eff_s": debug.get("eff_s"),
        "eff_gap": debug.get("eff_gap"),
        "c8_best": debug.get("c8_best"),
        "is_temp_like": debug.get("is_temp_like"),
    }

    # 最终标签对应的 C1~C8 分数（方便画图/过滤）
    detail = per_label.get(final_label, {})
    for k in ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]:
        row[f"{k}_score"] = detail.get(k, 0.0)

    return row



