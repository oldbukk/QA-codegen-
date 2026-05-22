"""
codegen_prompts.py — LLM-as-Coder 5 个阶段的 prompt 模板

阶段:
1. PLAN_SYSTEM / build_plan_user(schema)
2. CODEGEN_SYSTEM / build_codegen_user(strategy, schema)
3. REFINE_SYSTEM / build_refine_user(strategy, prev_code, error, items_sample)
4. REVIEW_SYSTEM / build_review_user(qa_samples)

所有 prompt 都强调：
- 严格基于 JSON records 的真实字段，不编造
- 输出格式严格（JSON 或纯代码）
- 函数签名：def gen_<name>(rows: list[dict], rng: random.Random) -> list[dict]
- 返回字典统一 schema: {"q":..., "a":..., "type":...} 或 {"messages":[...], "type": "multiturn"}
"""
from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# 基础约定（所有阶段都告知 LLM 的硬性规则）
# ---------------------------------------------------------------------------
QA_SCHEMA_SPEC = """\
QA 字典统一 schema:
  单轮：{"q": "<问题>", "a": "<答案>", "type": "<类型标签>"}
  多轮：{"messages": [{"role":"user"/"assistant","content":"..."} ...至少2轮交替...], "type": "multiturn"}

约束:
- 问题 q 长度 >= 6 字符；答案 a 长度 >= 8 字符且 <= 1500 字符
- 答案禁止出现：'无资料' '无意义' '未制定标准' '未制订标准' '尚不明确' '暂无' '未提供'
- 字段空值或上述占位符的，跳过不生成
- 所有事实必须来自 row 中实际字段值，禁止编造
- type 标签建议：single_field / multi_field / reasoning / scenario / summary / compare / multiturn
"""

ALLOWED_LIBS = """\
代码可用库：random, json, re, itertools, collections, math, string, copy, functools, operator, statistics
禁止使用：os, sys, subprocess, socket, shutil, open(), eval, exec, __import__, getattr, setattr, .__class__, .__bases__
"""


# ---------------------------------------------------------------------------
# Stage 1: Plan
# ---------------------------------------------------------------------------
REFERENCE_STRATEGY_CATALOG = """【参考策略目录（来自同项目规则版基线，171 行产出 87507 条 QA，每行约 500 条）】
你设计的策略**必须至少覆盖以下 5 大类**，每类下面列出的 qa_type 都是有效参考目标：

A. 基础事实/字段查询（kind=template，密度最高，每行 30+ 条）
   - single_field        每个非空字段对应一个"X 的 ___ 是什么？"。你的 template 必须能枚举所有相关字段，而不是只问 5 个。
   - alias_list          列出该化学品所有别名（若该行别名 == 主名则跳过，不得 emit 自指条目）
   - alias_resolve       给别名问主名（“二乙醒是什么的别名？"）— 仅在 alias != name 时 emit
   - translate_zh2en / translate_en2zh    中英名互翻 — 仅在 zh_name != en_name 且两者都非空时 emit
   - un_class            UN 编号 / 危险货物类别查询

B. 反向 / 判断 / 完形（kind=function，每行 10+ 条）
   - reverse_id          "给 CAS 号 / 分子式 / UN号 → 问是什么物质"
   - judge               "X 的沸点是否大于 100℃？" 是/否
   - cloze               填空式："___ 是甲醇的危险特性"
   - numeric_reverse     "沸点 78℃ 的常见溶剂是？"

C. 安全场景应急（kind=function，必须 ≥4 个场景策略，每行 5+ 条）
   - scenario_skin       问题描述“皮肤接触”场景 → 答案取自字段 **「皮肤接触」**（绝不允许用「健康危害」）
   - scenario_eye        问题描述“眼睛溅入”场景 → 答案取自字段 **「眼睛接触」**（绝不允许用「健康危害」）
   - scenario_inhale     问题描述“吸入”场景 → 答案取自字段 **「吸入」**
   - scenario_ingest     问题描述“误食”场景 → 答案取自字段 **「食入」**
   - scenario_leak       问题描述“泄漏”场景 → 答案取自字段 **「泄漏处理」**
   - scenario_fire       问题描述“火灾/灭火”场景 → 答案取自字段 **「消防措施」 或 「灭火方法」**
   - scenario_storage_risk  问题描述“储运/仓储风险”→ 答案取自字段 **「储运条件」 或 「禁忌物」**
   - 问题里必须含具体场景描述，不能只是“X 怎么急救？”

D. 比较 / 多跳推理（kind=function，每行 3-10 条）
   - compare_pair        两个化学品某参数比较（沸点/闪点等）
   - multihop_filter     "在含 N 的化学品中，UN 类别为 3 的有哪些？"
   - multihop_rank       "按沸点排序前 5"
   - multihop_water_react   "哪些遇水反应？"
   - multihop_flash_bucket  "闪点 < 0℃ 的化学品列表"

E. 综合 summary / 多轮（kind=function，每行 2-5 条）
   - summary             "请综合介绍 X" 多字段拼接长答案
   - combo_*             多字段组合查询（如同时给分子式和 CAS）
   - 多轮 messages：用户先问"X 是什么？" → 答 → 追问"那它的急救措施呢？" → 答

**不要只生成 A 类！** 规则版 87507 条里 A 类约 70%、B-E 类约 30%，但 B-E 类是训练价值最高的（场景化、推理类）。
"""

PLAN_SYSTEM = f"""你是一位资深数据工程师，专门为 SFT 训练集设计高质量 QA 生成策略。

你正在处理化学品 MSDS（材料安全数据表）数据。每行代表一个化学品，含约 50 个字段（中文名/分子式/物理参数/危险特性/防护/急救/储运...）。

你的任务：基于 schema 设计 **≥12 个互补**的 QA 生成策略，每个策略最终会被翻译成 Python 代码批量执行。

可调用工具探查数据真实情况，调够后调 finish_exploration 输出最终 JSON。

{REFERENCE_STRATEGY_CATALOG}

【硬性要求】
1. 总策略数 ≥ 12 个
2. C 类（场景应急）至少包含 4 个不同子场景（skin/eye/inhale/ingest/leak/fire/storage 选 ≥4 个）
3. A 类（template）密度要够大：必须列出 ≥10 个 target_fields，让生成器对每行每非空字段都产出一条 QA，预期每行 30+ 条
4. expected_per_row 必须给出真实的合理估计（A 类 ≥30，C 类 ≥3，其它 ≥1）

【输出格式】（finish_exploration 后下一条 assistant message）
严格输出 JSON 数组（无任何 markdown 装饰），每个对象字段：
{{
  "name": "短英文蛇形命名，对应参考目录里的 qa_type",
  "kind": "template" 或 "function",
  "target_fields": ["相关真实字段名列表（A 类 ≥10 个）"],
  "expected_per_row": 数字（每行预计产生 QA 条数，按上面"硬性要求 4"给）,
  "description": "20-60 字中文描述这个策略干什么、问什么、答什么"
}}

【kind 选择规则】
- template: 简单"问字段 → 答字段值"的固定问法（A 类基本全部用 template）。生成器只是渲染问句和答句模板，无需逻辑。
- function: 需要逻辑/比较/挑选/组合/多轮/场景化推理的策略（B/C/D/E 类必须 function）。

{QA_SCHEMA_SPEC}
"""


def build_plan_user(schema: dict, total_rows: int) -> str:
    """schema = {"columns": [...], "fill_rates": {col: rate}, "sample_rows": [...]}。"""
    cols = schema.get("columns", [])
    fill = schema.get("fill_rates", {})
    samples = schema.get("sample_rows", [])

    lines = [f"JSON records 共 {total_rows} 行，{len(cols)} 个字段。\n"]
    lines.append("【列名 + 非空率】")
    for c in cols:
        rate = fill.get(c, 0)
        lines.append(f"  - {c}  (非空率 {rate:.0%})")
    lines.append("\n【3 条示例行（已过滤空字段）】")
    for r in samples[:3]:
        lines.append(json.dumps(r, ensure_ascii=False)[:600])

    lines.append("\n请先调用工具核查 2-4 个你认为关键/不确定的字段，再调 finish_exploration，之后输出策略 JSON 数组。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 2: Codegen
# ---------------------------------------------------------------------------
CODEGEN_SYSTEM_FUNCTION = f"""你是 Python 工程师。请为给定策略编写一个生成函数。

【函数签名（必须严格一致）】
```python
def gen_{{strategy_name}}(rows, rng):
    \"\"\"rows: list[dict]，每个 dict 是 JSON record 清洗后的数据；rng: random.Random 实例。\"\"\"
    out = []
    # ... 逻辑 ...
    return out
```

{QA_SCHEMA_SPEC}

{ALLOWED_LIBS}

【硬性要求】
- 只输出函数源码（含必要的 import 在顶部），无 markdown 围栏、无解释文字、无 main 调用
- 字段空值判断：值为 "" / "无" / "无资料" / "无意义" / "未制定标准" / "未制订标准" / "尚不明确" / "暂无" / "未提供" / "-" / "—" 时视为空
- 每条 QA 必须包含 type 字段
- **每条 QA 必须包含 "source_row" 字段（取自 row["_行号"]）和 "chemical" 字段（取自 row["_主名"]），便于后续溯源审核**
- 跨行对比/反向推断类策略：source_row 填主要行号，可在 chemical 里用逗号串接多个名字
- 答案必须基于 row 实际字段值；禁止编造数据
- 化学品主名取 row["_主名"]（已预填）
- 函数返回 list[dict]，单次调用预期产出 {{expected_per_row}} × len(rows) 量级

【问句多样性 & 防崩塌硬要求】
1. **问句中必须出现本行的具体信息**，不允许全部行使用同一句泛化问话。至少满足以下任一：
   - 问句包含化学品名（row["_主名"]）
   - 问句包含本行另一个关键字段的具体值或片段（如 CAS 号、分子式、危险描述前 15-30 字等）
2. **反向推断/场景推断类策略**：必须把本行的危险描述/物理参数作为问题的主体提出，例如：
   - ✅ "某化学品具有以下危险特性：'极易燃烧爆炸。与空气混合能形成爆炸性混合物...' ，请推断它可能是哪类化学品？"
   - ❌ "根据危险特性和燃烧性，推断该化学品的可能特征或名称有哪些？"（泛化、不能汇该行）
3. **问句表达多样化**：为同一语义次位组造≥3种不同句式变体并随机抽用（避免所有行使用同一句模板）。句式变体示例：
   - "请问 X 的 Y 是什么？" / "X 的 Y 为何？" / "能告诉我 X 的 Y 吗？" / "关于 X，其 Y 是？"
4. **答案表达**：不要机械堆列，依据字段作专业、准确、完整的叙述。
4.1 **严禁答案泄题**：不要把目标字段的原文/片段塞进问句里再让 A 重复一遍。错误示例：
   - ❌ Q: "针对 二氧化碳，其皮肤接触部分描述为'若有冻伤，就医治疗。'，请问应采取何种措施？" A: "若有冻伤，就医治疗。"
   - ✅ Q: "工人皮肤溅到 二氧化碳，应如何处置？" A: "若有冻伤，就医治疗。"
   - 问句应只暴露"化学品名 + 场景/字段类型"，不暴露字段值本身。
5. **严禁自指/同义反复**：当 question 与 answer 本质问的就是化学品名时，必须先检查值是否等同于主名才决定是否 emit。否则会产出"乙炔的中文名是乙炔"这种垃圾条目。
   - alias_list / alias_resolve：若行里所有别名都等于主名，跳过该行不 emit
   - translate_zh2en / translate_en2zh：若中文名 == 英文名（或一方为空），跳过该行不 emit
   - 任何"X 的 N 是 X"的形式都必须避免
6. **场景类策略必须使用正确字段**：
   - scenario_eye → 字段 "眼睛接触"（不是 "健康危害"）
   - scenario_skin → 字段 "皮肤接触"
   - scenario_inhale → 字段 "吸入"
   - scenario_ingest → 字段 "食入"
   - scenario_leak → 字段 "泄漏处理"
   - scenario_fire → 字段 "消防措施" 或 "灭火方法"
   - scenario_storage_risk → 字段 "储运条件" 或 "禁忌物"
   - 错把"健康危害"塞给 scenario_eye/skin/inhale 等是常见且严重的错误，会被审核打回

可调用工具进一步查看数据，看完后调 finish_exploration 并输出最终代码.
"""

CODEGEN_SYSTEM_TEMPLATE = f"""你是数据工程师。请为给定的"template 类策略"输出一个 JSON 模板字典。

【模板字典 schema】
```json
{{
  "type_tag": "single_field_xxx",
  "items": [
    {{
      "field": "CAS号",
      "question_templates": ["{{name}} 的 CAS 号是多少？", "请告诉我 {{name}} 的 CAS 登记号。"],
      "answer_templates": ["{{name}} 的 CAS 号是 {{value}}。", "{{name}} 的 CAS 登记号为 {{value}}。"]
    }},
    ...更多字段...
  ]
}}
```

【占位符】
- {{name}}：化学品主名
- {{value}}：字段值
- 仅支持上述两个占位符

【硬性要求】
- **必须直接输出 JSON 字典字面量**（以 `{{` 开头，`}}` 结尾），无 markdown 围栏、无解释、无 Python 代码
- **严禁输出任何 Python 语句**：禁止 `def`、`import`、`return`、`for`、`if`、`row[...]`、`random.choice` 等代码片段
- 这是 template 策略，由 runner 负责按 {{name}}/{{value}} 占位符替换并遍历所有行；你只需提供模板字典
- field 必须是 JSON records 中的真实字段名
- **每个字段至少 4 个 question_templates + 4 个 answer_templates**，且句式要明显差异（疑问、疑问-礼貌、陈述-请你、疑问-口语等不同句式）
- type_tag 用蛇形英文
- 跳过空值由 runner 自动处理，不要在模板里写"无资料"等
- 若任务本质不适合 template（如需要跨行/条件分支/数值比较），请输出 `{{"items": []}}` 让 runner 跳过；不要硬塞 Python 代码

{QA_SCHEMA_SPEC}

可调用工具查看字段真实写法。
"""


def build_codegen_user(strategy: dict, schema: dict) -> str:
    """strategy 是 plan 阶段输出的单个对象。"""
    cols = schema.get("columns", [])
    target = strategy.get("target_fields", [])
    target_in = [c for c in target if c in cols]
    target_miss = [c for c in target if c not in cols]

    lines = [f"【策略名】{strategy['name']}",
             f"【kind】{strategy.get('kind', 'function')}",
             f"【描述】{strategy.get('description', '')}",
             f"【目标字段】{target_in}"]
    if target_miss:
        lines.append(f"【警告：以下字段在 JSON records 中不存在，请改用其他字段】{target_miss}")
    lines.append(f"【expected_per_row】{strategy.get('expected_per_row', 1)}")
    lines.append("\n请先核查 1-2 个字段的实际值分布（peek_column/stats_column），再调 finish_exploration 输出最终结果。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 3: Refine
# ---------------------------------------------------------------------------
REFINE_SYSTEM = f"""你的上一版代码失败了。请修复并重新输出完整代码（不要 diff，不要解释）。

{QA_SCHEMA_SPEC}

{ALLOWED_LIBS}

只输出修正后的完整函数源码，无 markdown 围栏。
"""


def build_refine_user(strategy_name: str, prev_code: str, error_msg: str,
                      n_produced: int, sample_items: list[dict]) -> str:
    sample_str = json.dumps(sample_items[:3], ensure_ascii=False, indent=2) if sample_items else "(无)"
    return f"""【策略】{strategy_name}

【上一版代码】
```python
{prev_code}
```

【失败原因】
{error_msg}

【产出数量】{n_produced}

【产出样例】
{sample_str}

【修复要点提示】
- 若失败原因提及“问句崩塌”或“多条使用相同问句”，请在问句中加入本行具体信息（化学品名 + 关键字段值片段），避免使用“该化学品”这种代词开头的泛化句式。
- 若失败原因提及“缺少 source_row/chemical”，请确保每条 QA dict 中都带 source_row 和 chemical。

请输出修复后的完整函数源码。
"""


# ---------------------------------------------------------------------------
# Stage 4: Review (QA 抽样质量审核)
# ---------------------------------------------------------------------------
REVIEW_SYSTEM = """你是 MSDS 化学品安全资深审核员。请逐条审核给定的 QA 对，判断答案是否：
1. 与提供的源数据字段一致（事实正确性）
2. 表达完整通顺
3. 不含"无资料/未提供/无意义"等占位符

【输出格式】严格输出 JSON 数组（无 markdown），每条：
{"idx": 输入序号, "keep": true/false, "reason": "10字内"}

只判断 keep/drop，不要修改 QA 内容。
"""


def build_review_user(samples: list[dict]) -> str:
    """samples: [{idx, q, a, source_row_data}]"""
    lines = ["请审核以下 QA："]
    for s in samples:
        lines.append(f"\n--- #{s['idx']} ---")
        lines.append(f"Q: {s['q']}")
        lines.append(f"A: {s['a']}")
        src = s.get("source_row_data", {})
        # 只取与 QA 可能相关的字段（限 1000 字）
        src_str = json.dumps(src, ensure_ascii=False)[:1200]
        lines.append(f"源行数据: {src_str}")
    lines.append("\n请输出 JSON 数组。")
    return "\n".join(lines)
