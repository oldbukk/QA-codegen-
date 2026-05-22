"""
codegen_tools.py — LLM agentic 数据探查工具

LLM 在 Plan / Codegen 阶段通过 OpenAI tools function calling
按需查看 JSON records 字段细节，避免一次性灌入全部数据。

四个工具：
- peek_column(col, n=10): 返回某列前 n 条非空原始值
- stats_column(col):      返回非空率、唯一值数、若可数则 min/max/mean
- peek_row(idx):          返回第 idx 行所有非空字段 dict
- sample_rows(n=3):       随机抽 n 行

公开接口：
- TOOL_SCHEMAS: list[dict]  → 喂给 client.chat.completions.create(tools=...)
- dispatch(name, args, rows, schema) -> dict   返回 JSON 友好结构
"""
from __future__ import annotations

import json
import random
import re
from typing import Any


# ---------------------------------------------------------------------------
# OpenAI tools schema
# ---------------------------------------------------------------------------
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "peek_column",
            "description": "查看 JSON records 某字段的前 n 条非空原始值。用于了解字段实际写法/长度/数值范围。",
            "parameters": {
                "type": "object",
                "properties": {
                    "col": {"type": "string", "description": "列名（中文或英文，需与 schema 完全一致）"},
                    "n": {"type": "integer", "description": "返回条数，默认 10，最多 30", "default": 10},
                },
                "required": ["col"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stats_column",
            "description": "统计某列：非空率、唯一值数；若值可解析为数值，给出 min/max/mean。",
            "parameters": {
                "type": "object",
                "properties": {
                    "col": {"type": "string"},
                },
                "required": ["col"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "peek_row",
            "description": "查看第 idx 行所有非空字段（已过滤空值/占位符）。idx 从 0 开始。",
            "parameters": {
                "type": "object",
                "properties": {
                    "idx": {"type": "integer"},
                },
                "required": ["idx"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sample_rows",
            "description": "随机抽 n 行完整数据（已过滤空值），用于观察数据分布。",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "default": 3},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_exploration",
            "description": "完成数据探查，输出最终结果（策略清单或代码）。调用此工具前请确认已有足够信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "对已观察数据的简短总结（10-50 字）"},
                },
                "required": ["summary"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")


def _parse_num(s: str) -> float | None:
    s = (s or "").strip().replace("，", ",").lstrip("约~≈").rstrip("℃%MPaKPakPaUPa ").strip()
    s = s.replace("−", "-").replace("－", "-")
    if _NUM_RE.match(s):
        try:
            return float(s)
        except Exception:
            return None
    return None


def dispatch(name: str, args: dict, rows: list[dict], seed: int = 0) -> dict:
    """执行一个工具调用，返回 JSON-serializable 字典。"""
    if name == "peek_column":
        col = str(args.get("col", "")).strip()
        n = int(args.get("n", 10))
        n = max(1, min(n, 30))
        if not rows or col not in rows[0] and col not in _all_keys(rows):
            return {"error": f"列名不存在: {col!r}"}
        values: list[str] = []
        for r in rows:
            v = str(r.get(col, "") or "").strip()
            if v and v not in {"无", "无资料", "无意义", "未制定标准", "未制订标准",
                               "尚不明确", "暂无", "未提供", "-", "—"}:
                values.append(v[:200])
                if len(values) >= n:
                    break
        return {"col": col, "values": values, "n_returned": len(values)}

    if name == "stats_column":
        col = str(args.get("col", "")).strip()
        if not rows:
            return {"error": "empty rows"}
        total = len(rows)
        non_empty = 0
        uniq: set[str] = set()
        nums: list[float] = []
        for r in rows:
            v = str(r.get(col, "") or "").strip()
            if v and v not in {"无", "无资料", "无意义", "未制定标准", "尚不明确"}:
                non_empty += 1
                uniq.add(v[:100])
                f = _parse_num(v)
                if f is not None:
                    nums.append(f)
        out: dict = {
            "col": col,
            "total": total,
            "non_empty": non_empty,
            "fill_rate": round(non_empty / total, 3) if total else 0,
            "unique_count": len(uniq),
        }
        if nums:
            out["numeric_min"] = min(nums)
            out["numeric_max"] = max(nums)
            out["numeric_mean"] = round(sum(nums) / len(nums), 4)
            out["numeric_count"] = len(nums)
        return out

    if name == "peek_row":
        idx = int(args.get("idx", 0))
        if idx < 0 or idx >= len(rows):
            return {"error": f"idx 越界 [0,{len(rows)-1}]"}
        r = rows[idx]
        out = {k: v for k, v in r.items()
               if not str(k).startswith("_") and str(v).strip()
               and str(v).strip() not in {"无", "无资料", "无意义", "未制定标准", "尚不明确"}}
        return {"idx": idx, "row": out}

    if name == "sample_rows":
        n = int(args.get("n", 3))
        n = max(1, min(n, 10))
        rng = random.Random(seed)
        sample = rng.sample(rows, min(n, len(rows)))
        out = []
        for r in sample:
            cleaned = {k: v for k, v in r.items()
                       if not str(k).startswith("_") and str(v).strip()
                       and str(v).strip() not in {"无", "无资料", "无意义"}}
            out.append({"_行号": r.get("_行号"), **cleaned})
        return {"n": len(out), "rows": out}

    if name == "finish_exploration":
        return {"finished": True, "summary": args.get("summary", "")}

    return {"error": f"unknown tool: {name}"}


def _all_keys(rows: list[dict]) -> set[str]:
    s: set[str] = set()
    for r in rows:
        s.update(r.keys())
    return s


# ---------------------------------------------------------------------------
# CLI 自检
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rows = [
        {"中文名": "乙炔", "CAS号": "74-86-2", "沸点（℃）": "-83.8", "_行号": 0},
        {"中文名": "苯", "CAS号": "71-43-2", "沸点（℃）": "80.1", "_行号": 1},
        {"中文名": "甲苯", "CAS号": "108-88-3", "沸点（℃）": "110.6", "_行号": 2},
    ]
    print(dispatch("peek_column", {"col": "CAS号", "n": 2}, rows))
    print(dispatch("stats_column", {"col": "沸点（℃）"}, rows))
    print(dispatch("peek_row", {"idx": 1}, rows))
    print(dispatch("sample_rows", {"n": 2}, rows, seed=1))
