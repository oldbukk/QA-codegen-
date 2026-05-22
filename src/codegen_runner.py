"""
codegen_runner.py — 模板字典渲染器 + 函数代码包装器

两种 kind 的统一接口：
- render_template(template_dict, rows, rng) -> list[QAItem]
- 函数 kind 的代码直接交给 codegen_sandbox.run_function_in_sandbox

公开接口：
- render_template(tpl: dict, rows: list[dict], rng: random.Random) -> list[dict]
- validate_qa_item(item: dict) -> dict | None    后置 QA 校验（长度/占位符）

模板字典示例：
{
  "type_tag": "single_field_basics",
  "items": [
    {"field": "CAS号",
     "question_templates": ["{name} 的 CAS 号是多少？"],
     "answer_templates": ["{name} 的 CAS 号是 {value}。"]}
  ]
}
"""
from __future__ import annotations

import random
import re as _re
from typing import Any

_NORM_PUNCT_RE = _re.compile(r"[\s\u3000，。？！,.\?\!；;：:、~～·\-—_/\\\(\)（）\[\]【】\"'`“”‘’《》<>]+")

EMPTY_TOKENS = {"", "无", "无资料", "无意义", "未制定标准", "未制订标准",
                "无标准", "-", "—", "尚不明确", "暂无", "未提供"}

BAD_TOKENS_IN_OUTPUT = ("无资料", "无意义", "未制定标准", "未制订标准",
                       "尚不明确", "暂无", "未提供")


def _is_empty(v: Any) -> bool:
    if v is None:
        return True
    return str(v).strip() in EMPTY_TOKENS


def render_template(tpl: dict, rows: list[dict], rng: random.Random,
                    variants_per_slot: int = 2) -> list[dict]:
    """渲染模板字典为 QA list。

    Args:
        tpl: {"type_tag": str, "items": [{"field", "question_templates", "answer_templates"}]}
        rows: 已清洗的行 dict
        rng: 随机源
        variants_per_slot: 每个 (row, slot) 产出几条不同问法（在 question_templates 中无放回抽取）

    Returns:
        list of {"q","a","type","chemical","source_row"}
    """
    type_tag = tpl.get("type_tag", "template_qa")
    items_spec = tpl.get("items", [])
    if not isinstance(items_spec, list):
        return []

    k_max = max(1, int(variants_per_slot))
    out: list[dict] = []
    for r in rows:
        name = r.get("_主名") or r.get("中文名") or ""
        if not name:
            continue
        for spec in items_spec:
            if not isinstance(spec, dict):
                continue
            field = spec.get("field", "")
            qts = spec.get("question_templates", []) or []
            ats = spec.get("answer_templates", []) or []
            if not field or not qts or not ats:
                continue
            value = r.get(field, "")
            if _is_empty(value):
                continue
            value_str = str(value).strip()
            # 在问句变体中无放回抽 k 条（不足则全取）
            k = min(k_max, len(qts))
            chosen_q = rng.sample(list(qts), k) if k < len(qts) else list(qts)
            for q_tpl in chosen_q:
                a_tpl = rng.choice(ats)
                try:
                    q = q_tpl.format(name=name, value=value_str)
                    a = a_tpl.format(name=name, value=value_str)
                except (KeyError, IndexError):
                    continue
                out.append({
                    "q": q, "a": a, "type": type_tag,
                    "chemical": name,
                    "source_row": r.get("_行号", -1),
                })
    return out


def validate_qa_item(item: dict) -> dict | None:
    """统一 QA 校验，返回清洗后的 item 或 None。

    保留原始 item 中的 chemical / source_row 元数据，供下游溯源审核使用。
    """
    if not isinstance(item, dict):
        return None
    meta = {
        "chemical": str(item.get("chemical", "")).strip(),
        "source_row": item.get("source_row", -1),
    }
    try:
        meta["source_row"] = int(meta["source_row"])
    except (TypeError, ValueError):
        meta["source_row"] = -1
    # 多轮
    if "messages" in item:
        msgs = item["messages"]
        if not isinstance(msgs, list) or len(msgs) < 2:
            return None
        cleaned = []
        for m in msgs:
            if not isinstance(m, dict):
                return None
            role = m.get("role")
            content = str(m.get("content", "")).strip()
            if role not in ("user", "assistant") or not content:
                return None
            if any(b in content for b in BAD_TOKENS_IN_OUTPUT):
                return None
            cleaned.append({"role": role, "content": content})
        for i, m in enumerate(cleaned):
            expect = "user" if i % 2 == 0 else "assistant"
            if m["role"] != expect:
                return None
        return {"messages": cleaned, "type": str(item.get("type", "multiturn")), **meta}

    # 单轮
    q = str(item.get("q", "")).strip()
    a = str(item.get("a", "")).strip()
    t = str(item.get("type", "")).strip() or "qa"
    if len(q) < 6 or len(a) < 8 or len(a) > 1500:
        return None
    if any(b in a for b in BAD_TOKENS_IN_OUTPUT):
        return None
    # 反自指过滤：答案除了化学品主名外几乎无新信息
    # e.g. "乙炔 中文名叫 乙炔"、"X 的英文名是 X"、"该别名 乙炔 的主名称是 乙炔"
    chem = meta.get("chemical", "")
    if chem and len(chem) >= 2:
        a_stripped = a.replace(chem, "").strip()
        # 去掉化学品名后剩 ≤ 4 个非空白字符（如"的 中 文 名 叫"）即判为自指
        a_residual = _NORM_PUNCT_RE.sub("", a_stripped)
        if len(a_residual) <= 4:
            return None
        # 答案完全等于主名（归一化后）
        if normalize_instruction(a) == normalize_instruction(chem):
            return None
        # 主名在答案中出现 ≥ 2 次，且去掉主名后残余无任何数字/字母信息
        # 通常意味着 "X 的 ___ 是 X" / "X 别名为 X" 这类同义反复
        if a.count(chem) >= 2 and not _re.search(r"[0-9A-Za-z]", a_stripped):
            return None
    return {"q": q, "a": a, "type": t, **meta}


# ---------------------------------------------------------------------------
# 公共工具：问句归一化 + 崩塌检测（供 build_qa_codegen 复用）
# ---------------------------------------------------------------------------


def normalize_instruction(text: str) -> str:
    """对问句做归一化用于去重与崩塌检测：去标点/空白、转小写。"""
    if not text:
        return ""
    s = str(text).lower()
    s = _NORM_PUNCT_RE.sub("", s)
    return s


def instruction_collapse_ratio(items: list[dict]) -> tuple[float, str]:
    """返回 (最大占比, 该占比对应的归一化问句) — 用于判断是否有泛化问句崩塌。

    占比 = 最多重复的同一归一化问句出现次数 / 单轮 item 总数。
    """
    if not items:
        return 0.0, ""
    from collections import Counter
    keys = []
    for it in items:
        q = it.get("q") or it.get("instruction") or ""
        nk = normalize_instruction(q)
        if nk:
            keys.append(nk)
    if not keys:
        return 0.0, ""
    c = Counter(keys)
    top_key, top_n = c.most_common(1)[0]
    return top_n / len(keys), top_key


# ---------------------------------------------------------------------------
# CLI 自检
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rows = [
        {"_主名": "乙炔", "中文名": "乙炔", "CAS号": "74-86-2", "分子式": "C2H2"},
        {"_主名": "苯", "中文名": "苯", "CAS号": "71-43-2", "分子式": "C6H6"},
        {"_主名": "无名", "中文名": "无名", "CAS号": "无", "分子式": "C6H6"},
    ]
    tpl = {
        "type_tag": "single_field",
        "items": [
            {"field": "CAS号",
             "question_templates": ["{name} 的 CAS 号是多少？", "请告诉我 {name} 的 CAS 登记号。"],
             "answer_templates": ["{name} 的 CAS 号是 {value}。"]},
            {"field": "分子式",
             "question_templates": ["{name} 的分子式？"],
             "answer_templates": ["{name} 的分子式为 {value}。"]},
        ]
    }
    rng = random.Random(0)
    items = render_template(tpl, rows, rng)
    print(f"Rendered {len(items)} QA items:")
    for it in items:
        print("  ", it)
        v = validate_qa_item(it)
        assert v is not None, f"validate failed: {it}"
