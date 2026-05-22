"""
build_qa_codegen.py — LLM-as-Coder QA 生成流水线

LLM 不直接产 QA，而是充当代码工程师：阅读 schema → 规划策略 →
（通过 OpenAI tools 函数调用按需查看真实数据）写 Python 代码 →
本地 AST + subprocess 沙箱验证 → 自愈 → 批处理全量执行 → LLM 抽样审核 → 输出。

设计要点（用户拍板：1C+1B fallback / 2C / 3A / 4C 强化版）：
- tools function calling 默认开启；不支持时降级为附 5 行样本到 prompt
- template / function 两种产物（混合）
- 全量阶段也走 subprocess 批处理（200 行/批），更安全
- LLM Review 抽 30 条做条级 keep/drop + 策略整体准确率监控

CLI：
  python build_qa_codegen.py \\
      --input-json 抽取结果.json --model gpt-4.1-mini \\
      --base-url https://api.openai-proxy.org/v1 --api-key sk-xxx \\
      --max-rows 0 --workers 4 --batch-size 200 \\
      --review-per-strategy 30 --review-threshold 0.6

输出：
  schema.json                        Inspect 产物
  plan.json                          Plan 产物
  generated_generators/<name>.py     函数代码
  generated_templates/<name>.json    模板字典
  codegen_logs/*.jsonl               全部 LLM 对话 + 沙箱错误
  cache_codegen/<sha1>.json          LLM 缓存
  qa_codegen.alpaca.jsonl
  qa_codegen.chatml.jsonl
  qa_codegen.stats.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI, APIError, APITimeoutError, RateLimitError

from codegen_prompts import (
    PLAN_SYSTEM, build_plan_user,
    CODEGEN_SYSTEM_FUNCTION, CODEGEN_SYSTEM_TEMPLATE, build_codegen_user,
    REFINE_SYSTEM, build_refine_user,
    REVIEW_SYSTEM, build_review_user,
)
from codegen_tools import TOOL_SCHEMAS, dispatch as tool_dispatch
from codegen_sandbox import run_function_in_sandbox, check_ast
from codegen_runner import (render_template, validate_qa_item, EMPTY_TOKENS,
                            normalize_instruction, instruction_collapse_ratio)

SYSTEM_PROMPT_FOR_DATASET = (
    "你是化学品安全应急助手，依据 MSDS 资料严谨、专业地作答；"
    "当资料未提供相关信息时，应明确告知不掌握。"
)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class Config:
    input_json: str = "抽取结果.json"
    out_prefix: str = "qa_codegen"
    model: str = "gpt-4.1-mini"
    base_url: str = ""
    api_key: str = ""
    workers: int = 4
    temperature: float = 0.4
    max_tokens: int = 4096
    max_retries: int = 3
    retry_base: float = 2.0
    timeout: float = 120.0
    cache_dir: str = "cache_codegen"
    log_dir: str = "codegen_logs"
    gen_func_dir: str = "generated_generators"
    gen_tpl_dir: str = "generated_templates"
    use_tools: bool = True
    tool_loop_max: int = 6
    refine_max: int = 3
    batch_size: int = 200
    sandbox_sample_rows: int = 10
    sandbox_timeout: int = 30
    sandbox_mem_mb: int = 512
    review_per_strategy: int = 30
    review_threshold: float = 0.3  # < 阈值 → 丢弃整策略；否则条级过滤即可
    max_rows: int = 0
    seed: int = 20260515
    strategies_override: str = ""  # 逗号分隔，限制只跑指定 plan 策略
    variants_per_slot: int = 1   # 每 (row, slot) 抽 K 个问句变体（>1 会产生同答案多问句的训练对）
    collapse_max_ratio: float = 0.5  # 某策略中同一归一化问句占比 > 阈值 → 判为问句崩塌


# ---------------------------------------------------------------------------
# JSON 加载（复用 build_qa_llm.py 清洗逻辑）
# ---------------------------------------------------------------------------
def _is_empty(v: Any) -> bool:
    return v is None or str(v).strip() in EMPTY_TOKENS


def _clean_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).replace("\u3000", " ").strip()
    s = re.sub(r"[ \t]+", " ", s)
    return s


def _split_names(s: str) -> list[str]:
    if not s:
        return []
    parts = re.split(r"[；;、]", s)
    return [p.strip() for p in parts if p.strip()]


def _load_json_records(input_json_path: str) -> list[dict]:
    path = Path(input_json_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        records = data.get("records")
    else:
        records = data
    if not isinstance(records, list):
        raise ValueError("input JSON must be a list or an object with a records list")
    if not all(isinstance(item, dict) for item in records):
        raise ValueError("input JSON records must be objects")
    return records


def load_rows_from_records(records: list[dict]) -> list[dict]:
    all_columns: list[str] = []
    seen_columns: set[str] = set()
    for raw in records:
        for key in raw:
            key = str(key).lstrip("\ufeff")
            if key not in seen_columns:
                seen_columns.add(key)
                all_columns.append(key)

    vap_cols = [c for c in all_columns if "饱和蒸汽压" in c]
    rows: list[dict] = []
    for i, raw in enumerate(records):
        r = {str(k).lstrip("\ufeff"): _clean_text(v) for k, v in raw.items()}
        if len(vap_cols) > 1:
            r["饱和蒸汽压（kPa）"] = next(
                (_clean_text(r.get(c, "")) for c in vap_cols if not _is_empty(r.get(c, ""))),
                "",
            )
            for c in vap_cols:
                if c != "饱和蒸汽压（kPa）":
                    r.pop(c, None)

        name_fields = [c for c in r if "名称" in c or c.endswith("名") or c == "中文名" or c == "英文名"]
        all_names: list[str] = []
        for c in name_fields:
            for n in _split_names(r.get(c, "")):
                if n and n not in all_names:
                    all_names.append(n)
        if not all_names:
            continue
        r["_主名"] = all_names[0]
        r["_别名"] = all_names
        r["_行号"] = int(i)
        r = {k: v for k, v in r.items() if k.startswith("_") or not _is_empty(v)}
        rows.append(r)
    return rows


def load_rows(input_json_path: str) -> list[dict]:
    return load_rows_from_records(_load_json_records(input_json_path))


def build_schema(rows: list[dict], sample_n: int = 3, seed: int = 0) -> dict:
    """从 rows 计算 schema：列名 + 非空率 + sample_n 行示例。"""
    cols_seen: dict[str, int] = defaultdict(int)
    for r in rows:
        for k, v in r.items():
            if str(k).startswith("_"):
                continue
            if not _is_empty(v):
                cols_seen[k] += 1
    total = len(rows) or 1
    cols = list(cols_seen.keys())
    fill_rates = {c: cols_seen[c] / total for c in cols}

    rng = random.Random(seed)
    samples = []
    for r in rng.sample(rows, min(sample_n, len(rows))):
        cleaned = {k: v for k, v in r.items()
                   if not str(k).startswith("_") and not _is_empty(v)}
        cleaned["_主名"] = r.get("_主名", "")
        cleaned["_行号"] = r.get("_行号", -1)
        samples.append(cleaned)

    return {"columns": cols, "fill_rates": fill_rates,
            "sample_rows": samples, "total_rows": total}


# ---------------------------------------------------------------------------
# LLM 客户端（含缓存、tool calling 循环）
# ---------------------------------------------------------------------------
class LLMClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        kwargs: dict[str, Any] = {}
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        self.client = OpenAI(**kwargs)
        self.cache_dir = Path(cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tool_call_supported: Optional[bool] = None  # None=未检测

    def _cache_key(self, payload: dict) -> Path:
        h = hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        return self.cache_dir / f"{h}.json"

    def chat_with_tools(
        self,
        system: str,
        user: str,
        tool_context_rows: list[dict],
        cache_tag: str = "",
    ) -> tuple[str, list[dict]]:
        """带 tool calling 的对话循环。

        返回 (最终 assistant text, 工具调用日志)。
        """
        cache_payload = {
            "tag": cache_tag,
            "model": self.cfg.model,
            "system": system,
            "user": user,
            "temperature": self.cfg.temperature,
            "tools": bool(self.cfg.use_tools),
        }
        cache_path = self._cache_key(cache_payload)
        if cache_path.exists():
            try:
                obj = json.loads(cache_path.read_text(encoding="utf-8"))
                return obj["content"], obj.get("tool_log", [])
            except Exception:
                pass

        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        tool_log: list[dict] = []

        use_tools = self.cfg.use_tools and self.tool_call_supported is not False

        for loop_i in range(self.cfg.tool_loop_max + 1):
            resp = self._safe_chat(messages, use_tools=use_tools)
            if resp is None:
                return "", tool_log
            msg = resp.choices[0].message

            # 没有 tool call 或不支持 → 终止
            tool_calls = getattr(msg, "tool_calls", None) or []
            if not tool_calls or not use_tools:
                content = msg.content or ""
                cache_path.write_text(
                    json.dumps({"content": content, "tool_log": tool_log},
                               ensure_ascii=False),
                    encoding="utf-8",
                )
                return content, tool_log

            # 模型确实在使用 tools，标记支持
            self.tool_call_supported = True

            # 处理 tool calls
            assistant_msg = {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    } for tc in tool_calls
                ],
            }
            messages.append(assistant_msg)

            finished = False
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                result = tool_dispatch(name, args, tool_context_rows, seed=self.cfg.seed + loop_i)
                tool_log.append({"loop": loop_i, "name": name, "args": args,
                                 "result_preview": json.dumps(result, ensure_ascii=False)[:300]})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False)[:4000],
                })
                if name == "finish_exploration":
                    finished = True

            if finished:
                # 再追一轮让模型产出最终内容
                resp = self._safe_chat(messages, use_tools=False)
                if resp is None:
                    return "", tool_log
                content = resp.choices[0].message.content or ""
                cache_path.write_text(
                    json.dumps({"content": content, "tool_log": tool_log},
                               ensure_ascii=False),
                    encoding="utf-8",
                )
                return content, tool_log

        # 达到 tool_loop_max，仍未 finish → 强制再调一次让其总结
        messages.append({
            "role": "user",
            "content": "已达工具调用上限，请立即输出最终结果（不再调用工具）。",
        })
        resp = self._safe_chat(messages, use_tools=False)
        content = (resp.choices[0].message.content if resp else "") or ""
        cache_path.write_text(
            json.dumps({"content": content, "tool_log": tool_log},
                       ensure_ascii=False),
            encoding="utf-8",
        )
        return content, tool_log

    def chat_simple(self, system: str, user: str, cache_tag: str = "") -> str:
        """无 tools 调用的简单对话。"""
        cache_payload = {
            "tag": cache_tag,
            "model": self.cfg.model,
            "system": system,
            "user": user,
            "temperature": self.cfg.temperature,
        }
        cache_path = self._cache_key(cache_payload)
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))["content"]
            except Exception:
                pass

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        resp = self._safe_chat(messages, use_tools=False)
        content = (resp.choices[0].message.content if resp else "") or ""
        cache_path.write_text(
            json.dumps({"content": content}, ensure_ascii=False),
            encoding="utf-8",
        )
        return content

    def _safe_chat(self, messages: list[dict], use_tools: bool):
        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            try:
                kwargs: dict[str, Any] = dict(
                    model=self.cfg.model,
                    messages=messages,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    timeout=self.cfg.timeout,
                )
                if use_tools:
                    kwargs["tools"] = TOOL_SCHEMAS
                    kwargs["tool_choice"] = "auto"
                return self.client.chat.completions.create(**kwargs)
            except (RateLimitError, APITimeoutError, APIError) as e:
                last_err = e
                msg = str(e)
                # 检测：模型不支持 tools
                if "tool" in msg.lower() and use_tools:
                    print(f"  [tools-unsupported] {msg[:200]}; falling back to plain chat")
                    self.tool_call_supported = False
                    use_tools = False
                    continue
                sleep = self.cfg.retry_base ** attempt + random.uniform(0, 1)
                print(f"  [retry {attempt+1}/{self.cfg.max_retries}] {type(e).__name__}; sleep {sleep:.1f}s")
                time.sleep(sleep)
            except Exception as e:
                last_err = e
                print(f"  [error] {type(e).__name__}: {e}")
                break
        print(f"  [give-up] {last_err}")
        return None


# ---------------------------------------------------------------------------
# Phase 1: Inspect — 写 schema.json
# ---------------------------------------------------------------------------
def phase_inspect(cfg: Config, rows: list[dict]) -> dict:
    print(f"[Phase 1] Inspect — {len(rows)} 行, 计算 schema ...")
    schema = build_schema(rows, sample_n=5, seed=cfg.seed)
    with open("schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    print(f"  → schema.json  ({len(schema['columns'])} 列)")
    return schema


# ---------------------------------------------------------------------------
# Phase 2: Plan — LLM 输出策略 JSON 数组
# ---------------------------------------------------------------------------
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)```", re.IGNORECASE)


def _extract_json(text: str, want: str = "array") -> Any:
    """提取 JSON：先剥 markdown 围栏，再找首个 [..] / {..}。"""
    if not text:
        return None
    s = text.strip()
    candidates: list[str] = []
    # 1) 直接
    candidates.append(s)
    # 2) markdown
    m = _JSON_BLOCK_RE.search(s)
    if m:
        candidates.append(m.group(1).strip())
    # 3) 找首个 [/{ 到末尾 ]/}
    if want == "array":
        m = re.search(r"\[[\s\S]+\]", s)
    else:
        m = re.search(r"\{[\s\S]+\}", s)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            continue
    return None


def phase_plan(cfg: Config, llm: LLMClient, schema: dict, rows: list[dict]) -> list[dict]:
    print(f"[Phase 2] Plan — LLM 设计策略 ...")
    user = build_plan_user(schema, total_rows=len(rows))
    text, tool_log = llm.chat_with_tools(
        PLAN_SYSTEM, user,
        tool_context_rows=rows,
        cache_tag="plan",
    )
    _log(cfg, "plan", {"text": text, "tool_log": tool_log})

    plans = _extract_json(text, want="array")
    if not isinstance(plans, list) or not plans:
        print(f"  [ERROR] plan 解析失败，输出预览：{text[:400]}")
        return []

    # 校验/清理
    valid = []
    for p in plans:
        if not isinstance(p, dict):
            continue
        name = re.sub(r"[^a-z0-9_]+", "_", str(p.get("name", "")).lower()).strip("_")
        kind = p.get("kind", "function")
        if kind not in ("template", "function"):
            kind = "function"
        target = [c for c in p.get("target_fields", []) if isinstance(c, str)]
        if not name or not target:
            continue
        valid.append({
            "name": name,
            "kind": kind,
            "target_fields": target,
            "expected_per_row": int(p.get("expected_per_row", 1)),
            "description": str(p.get("description", ""))[:200],
        })

    with open("plan.json", "w", encoding="utf-8") as f:
        json.dump(valid, f, ensure_ascii=False, indent=2)
    print(f"  → plan.json — {len(valid)} 个策略：{[p['name'] for p in valid]}")
    return valid


# ---------------------------------------------------------------------------
# Phase 3: Codegen — 为每个策略生成代码或模板
# ---------------------------------------------------------------------------
def _strip_code_fence(s: str) -> str:
    """去掉 ```python ... ``` 围栏，返回纯代码。"""
    s = s.strip()
    m = re.search(r"```(?:python)?\s*([\s\S]+?)```", s, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return s


def codegen_one_strategy(
    cfg: Config, llm: LLMClient, strategy: dict, schema: dict, rows: list[dict],
) -> dict:
    """单个策略：生成 → 沙箱验证 → refine 至多 N 次 → 落盘。

    Returns:
        {"name", "kind", "ok", "n_sample", "path", "error", ...}
    """
    name = strategy["name"]
    kind = strategy["kind"]
    sys_prompt = (CODEGEN_SYSTEM_TEMPLATE if kind == "template"
                  else CODEGEN_SYSTEM_FUNCTION.replace("{strategy_name}", name))
    user_prompt = build_codegen_user(strategy, schema)
    sample_rows = rows[:cfg.sandbox_sample_rows]

    last_artifact = ""
    last_error = ""
    last_items: list[dict] = []

    for attempt in range(cfg.refine_max + 1):
        if attempt == 0:
            text, tool_log = llm.chat_with_tools(
                sys_prompt, user_prompt,
                tool_context_rows=rows,
                cache_tag=f"codegen:{name}",
            )
        else:
            refine_user = build_refine_user(name, last_artifact, last_error, len(last_items), last_items)
            text = llm.chat_simple(REFINE_SYSTEM, refine_user,
                                   cache_tag=f"refine:{name}:{attempt}")
            tool_log = []

        if not text:
            last_error = "LLM 空响应"
            continue

        _log(cfg, f"codegen:{name}:try{attempt}",
             {"sys_len": len(sys_prompt), "user_len": len(user_prompt),
              "text": text, "tool_log": tool_log})

        if kind == "template":
            artifact = _extract_json(text, want="object")
            if not isinstance(artifact, dict) or "items" not in artifact:
                last_error = f"模板 JSON 解析失败：{text[:300]}"
                last_artifact = text
                continue
            # 模板渲染 + 校验
            rng = random.Random(cfg.seed)
            items = render_template(artifact, sample_rows, rng,
                                    variants_per_slot=cfg.variants_per_slot)
            items = [v for v in (validate_qa_item(x) for x in items) if v]
            if not items:
                last_error = "模板渲染后无有效 QA（可能字段名错或值全为空）"
                last_artifact = json.dumps(artifact, ensure_ascii=False)
                last_items = []
                continue
            # 落盘
            path = Path(cfg.gen_tpl_dir) / f"{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
            return {"name": name, "kind": kind, "ok": True,
                    "path": str(path), "n_sample": len(items),
                    "sample_items": items[:3]}
        else:
            code = _strip_code_fence(text)
            # AST 检查
            ast_ok, ast_msg = check_ast(code)
            if not ast_ok:
                last_error = f"AST 拒绝: {ast_msg}"
                last_artifact = code
                last_items = []
                continue
            # 必须含目标函数
            func_name = f"gen_{name}"
            if f"def {func_name}" not in code:
                # 也允许函数名首字符不同（容错）
                m = re.search(r"def\s+(gen_\w+)\s*\(", code)
                if m:
                    func_name = m.group(1)
                else:
                    last_error = f"未找到 gen_* 函数定义"
                    last_artifact = code
                    last_items = []
                    continue
            # 沙箱跑样本
            r = run_function_in_sandbox(
                code, func_name, sample_rows,
                seed=cfg.seed,
                timeout=cfg.sandbox_timeout,
                mem_mb=cfg.sandbox_mem_mb,
            )
            if not r["ok"]:
                last_error = f"沙箱失败: {r['error']}"
                last_artifact = code
                last_items = []
                continue
            # QA 后置校验
            items = [v for v in (validate_qa_item(x) for x in r["items"]) if v]
            if not items:
                last_error = f"产出 {r['n']} 条但全部不通过 QA 校验"
                last_artifact = code
                last_items = r["items"][:3]
                continue
            # 通过！
            path = Path(cfg.gen_func_dir) / f"{name}.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            # 保存元数据 + 代码
            header = (f"# Generated by build_qa_codegen.py\n"
                      f"# strategy: {name}\n"
                      f"# func_name: {func_name}\n"
                      f"# expected_per_row: {strategy.get('expected_per_row', 1)}\n\n")
            path.write_text(header + code, encoding="utf-8")
            return {"name": name, "kind": kind, "ok": True,
                    "path": str(path), "func_name": func_name,
                    "n_sample": len(items), "sample_items": items[:3]}

    return {"name": name, "kind": kind, "ok": False,
            "error": last_error, "last_artifact": last_artifact[:1000]}


def phase_codegen(cfg: Config, llm: LLMClient, plans: list[dict],
                  schema: dict, rows: list[dict]) -> list[dict]:
    print(f"[Phase 3] Codegen — {len(plans)} 策略 (workers={cfg.workers}) ...")
    Path(cfg.gen_func_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.gen_tpl_dir).mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    if cfg.workers <= 1:
        for p in plans:
            r = codegen_one_strategy(cfg, llm, p, schema, rows)
            print(f"  [{r['kind']}] {r['name']}: ok={r['ok']} "
                  f"{'n_sample=' + str(r.get('n_sample',0)) if r['ok'] else r.get('error','')[:120]}")
            results.append(r)
    else:
        with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
            futs = {ex.submit(codegen_one_strategy, cfg, llm, p, schema, rows): p["name"]
                    for p in plans}
            for fut in as_completed(futs):
                r = fut.result()
                print(f"  [{r['kind']}] {r['name']}: ok={r['ok']} "
                      f"{'n_sample=' + str(r.get('n_sample',0)) if r['ok'] else r.get('error','')[:120]}")
                results.append(r)
    return results


# ---------------------------------------------------------------------------
# Phase 4: Full Run (subprocess 批处理)
# ---------------------------------------------------------------------------
def run_function_full(cfg: Config, code: str, func_name: str,
                       rows: list[dict]) -> list[dict]:
    """通过沙箱分批跑全量，合并 items。"""
    all_items: list[dict] = []
    for i in range(0, len(rows), cfg.batch_size):
        batch = rows[i:i + cfg.batch_size]
        r = run_function_in_sandbox(
            code, func_name, batch,
            seed=cfg.seed + i,
            timeout=max(cfg.sandbox_timeout, 60),
            mem_mb=cfg.sandbox_mem_mb,
        )
        if not r["ok"]:
            print(f"    [batch {i}-{i+len(batch)}] FAIL: {r['error']}")
            continue
        all_items.extend(r["items"])
    return all_items


def phase_full_run(cfg: Config, codegen_results: list[dict],
                   rows: list[dict]) -> dict[str, list[dict]]:
    """对每个通过的策略跑全量数据，返回 {strategy_name: [qa_items]}。

    保留 raw items 上的 chemical/source_row 元数据，不在此处做 validate（交给 attach_meta）。
    增加问句崩塌门控：若策略中同一归一化问句占比 > collapse_max_ratio，丢弃该策略。
    """
    print(f"[Phase 4] Full Run — 批量 {cfg.batch_size}/批 ...")
    out: dict[str, list[dict]] = {}
    for r in codegen_results:
        if not r["ok"]:
            continue
        name = r["name"]
        if r["kind"] == "template":
            tpl = json.loads(Path(r["path"]).read_text(encoding="utf-8"))
            rng = random.Random(cfg.seed)
            items = render_template(tpl, rows, rng,
                                    variants_per_slot=cfg.variants_per_slot)
        else:
            code = Path(r["path"]).read_text(encoding="utf-8")
            items = run_function_full(cfg, code, r["func_name"], rows)
        # 问句崩塌门控（基于 raw items，单轮）
        single_for_check = [x for x in items
                            if isinstance(x, dict) and "q" in x and "messages" not in x]
        ratio, top_key = instruction_collapse_ratio(single_for_check)
        if ratio > cfg.collapse_max_ratio and len(single_for_check) >= 4:
            print(f"  {name}: ⚠️  问句崩塌 {ratio:.0%} 行使用同一问句 → 整策略丢弃")
            print(f"     崩塌问句样例: {top_key[:80]}")
            continue
        out[name] = items
        print(f"  {name}: {len(items)} 条" +
              (f"（最高重复问句 {ratio:.0%}）" if ratio > 0.2 else ""))
    return out


# ---------------------------------------------------------------------------
# Phase 5: LLM Review — 抽样审核
# ---------------------------------------------------------------------------
def phase_review(cfg: Config, llm: LLMClient,
                 qa_by_strategy: dict[str, list[dict]],
                 rows: list[dict]) -> dict[str, list[dict]]:
    """对每策略抽 N 条做 LLM 评分，drop 失败条 + 整策略准确率 < threshold 则丢弃整策略。"""
    print(f"[Phase 5] LLM Review — 抽样 {cfg.review_per_strategy}/策略, "
          f"阈值 {cfg.review_threshold} ...")
    row_by_id = {r["_行号"]: r for r in rows}
    rng = random.Random(cfg.seed + 999)
    filtered: dict[str, list[dict]] = {}
    review_stats: dict[str, dict] = {}

    for name, items in qa_by_strategy.items():
        if not items:
            review_stats[name] = {"sampled": 0, "kept": 0, "rate": 0, "dropped_strategy": False}
            continue
        # 多轮 / 单轮分开抽样；items 已是 normalized 格式（含 instruction/output 或 messages）
        single = [it for it in items if "instruction" in it]
        sample_n = min(cfg.review_per_strategy, len(single))
        if sample_n == 0:
            filtered[name] = items
            review_stats[name] = {"sampled": 0, "kept": len(items), "rate": 1.0,
                                  "dropped_strategy": False, "note": "no single QA to review"}
            continue
        idx_sample = rng.sample(range(len(single)), sample_n)
        review_input = []
        for i, ix in enumerate(idx_sample):
            it = single[ix]
            src_row = row_by_id.get(it.get("source_row", -1), {})
            src_data = {k: v for k, v in src_row.items()
                        if not str(k).startswith("_") and not _is_empty(v)}
            review_input.append({
                "idx": i,
                "q": it["instruction"],
                "a": it["output"],
                "source_row_data": src_data,
            })
        user = build_review_user(review_input)
        text = llm.chat_simple(REVIEW_SYSTEM, user, cache_tag=f"review:{name}")
        verdicts = _extract_json(text, want="array") or []
        keep_map: dict[int, bool] = {}
        drop_reasons: list[str] = []
        for v in verdicts:
            if isinstance(v, dict) and "idx" in v:
                keep_map[int(v["idx"])] = bool(v.get("keep", True))
                if not v.get("keep", True):
                    drop_reasons.append(str(v.get("reason", ""))[:30])

        kept_in_sample = sum(1 for i in range(sample_n) if keep_map.get(i, True))
        rate = kept_in_sample / sample_n if sample_n else 1.0

        if rate < cfg.review_threshold:
            print(f"  [{name}] 抽样准确率 {rate:.0%} < 阈值 {cfg.review_threshold:.0%} "
                  f"→ 整策略丢弃；reasons={drop_reasons[:5]}")
            review_stats[name] = {"sampled": sample_n, "kept": kept_in_sample,
                                  "rate": rate, "dropped_strategy": True,
                                  "reasons": drop_reasons[:10]}
            continue

        # 条级过滤：被 review 拒绝的样本去掉，其他保留
        rejected_real_idx = {idx_sample[i] for i, kept in keep_map.items() if not kept}
        kept_single = [it for i, it in enumerate(single) if i not in rejected_real_idx]
        non_single = [it for it in items if "instruction" not in it]
        filtered[name] = kept_single + non_single

        review_stats[name] = {"sampled": sample_n, "kept": kept_in_sample,
                              "rate": rate, "dropped_strategy": False,
                              "items_dropped_by_review": len(rejected_real_idx)}
        print(f"  [{name}] 抽样 {sample_n}, 通过 {kept_in_sample} ({rate:.0%}) → "
              f"输出 {len(filtered[name])} 条")

    return filtered, review_stats  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Phase 6: Post-process + Write
# ---------------------------------------------------------------------------
def dedup(items: list[dict]) -> list[dict]:
    """两轮去重：
    1. 严格重复：同 (qa_type, 归一化 instruction, 归一化 output)。
    2. 语义重复：同 (chemical, source_row, qa_type, 归一化 output)
       → 同一化学品同一行同一类型下答案一致的只保留 1 条，消除“同答案多问法”凗余。
    """
    seen_strict: set = set()
    seen_semantic: set = set()
    out: list[dict] = []
    for it in items:
        if "messages" in it:
            key = ("MT", it.get("chemical", ""),
                   json.dumps(it["messages"], ensure_ascii=False))
            if key in seen_strict:
                continue
            seen_strict.add(key)
            out.append(it)
            continue
        inst_norm = normalize_instruction(it.get("instruction", ""))
        out_norm = normalize_instruction(it.get("output", ""))[:200]
        qa_type = it.get("qa_type", "")
        strict_key = (qa_type, inst_norm, out_norm)
        if strict_key in seen_strict:
            continue
        semantic_key = (it.get("chemical", ""), it.get("source_row", -1),
                        qa_type, out_norm)
        if semantic_key in seen_semantic:
            continue
        seen_strict.add(strict_key)
        seen_semantic.add(semantic_key)
        out.append(it)
    return out


def to_alpaca_record(it: dict) -> dict:
    if "messages" in it:
        msgs = it["messages"]
        prev = "\n".join(f"{m['role']}: {m['content']}" for m in msgs[:-2]) if len(msgs) > 2 else ""
        return {
            "instruction": msgs[-2]["content"],
            "input": f"前文对话：\n{prev}" if prev else "",
            "output": msgs[-1]["content"],
            "qa_type": it["qa_type"],
            "chemical": it.get("chemical", ""),
            "source_row": it.get("source_row", -1),
            "strategy": it.get("strategy", ""),
        }
    return {
        "instruction": it["instruction"],
        "input": it.get("input", ""),
        "output": it["output"],
        "qa_type": it["qa_type"],
        "chemical": it.get("chemical", ""),
        "source_row": it.get("source_row", -1),
        "strategy": it.get("strategy", ""),
    }


def to_chatml_record(it: dict) -> dict:
    if "messages" in it:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT_FOR_DATASET}] + it["messages"]
    else:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT_FOR_DATASET},
            {"role": "user", "content": it["instruction"]},
            {"role": "assistant", "content": it["output"]},
        ]
    return {
        "messages": msgs,
        "qa_type": it["qa_type"],
        "chemical": it.get("chemical", ""),
        "source_row": it.get("source_row", -1),
        "strategy": it.get("strategy", ""),
    }


def attach_meta(qa_items: list[dict], strategy: str, rows_by_id: dict[int, dict]) -> list[dict]:
    """把策略输出的 raw items 转为带 instruction/output/source_row/chemical 的统一格式。"""
    out = []
    for it in qa_items:
        v = validate_qa_item(it)
        if v is None:
            continue
        # raw item 可能含 source_row / chemical 字段（LLM 写的代码可填）
        source_row = it.get("source_row", -1)
        try:
            source_row = int(source_row)
        except (TypeError, ValueError):
            source_row = -1
        chemical = it.get("chemical", "")
        if source_row >= 0 and source_row in rows_by_id and not chemical:
            chemical = rows_by_id[source_row].get("_主名", "")
        qa_type = v.get("type", "qa")
        if "messages" in v:
            out.append({
                "qa_type": qa_type,
                "messages": v["messages"],
                "chemical": chemical,
                "source_row": source_row,
                "strategy": strategy,
            })
        else:
            out.append({
                "qa_type": qa_type,
                "instruction": v["q"],
                "input": "",
                "output": v["a"],
                "chemical": chemical,
                "source_row": source_row,
                "strategy": strategy,
            })
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _log(cfg: Config, tag: str, obj: dict) -> None:
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    path = Path(cfg.log_dir) / f"{tag.replace(':','_').replace('/','_')}.json"
    try:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)[:200_000], encoding="utf-8")
    except Exception:
        pass


def main(cfg: Config) -> None:
    random.seed(cfg.seed)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

    print(f"[Config] model={cfg.model}  base_url={cfg.base_url or '(default)'}")
    print(f"[Config] workers={cfg.workers}  batch={cfg.batch_size}  use_tools={cfg.use_tools}")

    rows = load_rows(cfg.input_json)
    if cfg.max_rows and cfg.max_rows > 0:
        rows = rows[:cfg.max_rows]
    print(f"[Data] 载入 {len(rows)} 行")

    rows_by_id = {r["_行号"]: r for r in rows}

    # Phase 1
    schema = phase_inspect(cfg, rows)

    # LLM 客户端
    llm = LLMClient(cfg)

    # Phase 2
    plans = phase_plan(cfg, llm, schema, rows)
    if not plans:
        print("[FATAL] 无可用策略，退出")
        return

    if cfg.strategies_override:
        keep = set(cfg.strategies_override.split(","))
        plans = [p for p in plans if p["name"] in keep]
        print(f"[Override] 过滤后 {len(plans)} 策略")

    # Phase 3
    codegen_results = phase_codegen(cfg, llm, plans, schema, rows)
    success = [r for r in codegen_results if r["ok"]]
    print(f"[Phase 3 Done] 成功 {len(success)}/{len(codegen_results)} 策略")
    if not success:
        print("[FATAL] 全部策略代码生成失败，退出")
        return

    # Phase 4
    raw_by_strategy = phase_full_run(cfg, success, rows)

    # 转为统一格式（含 instruction/output/source_row/chemical）
    qa_by_strategy: dict[str, list[dict]] = {}
    for name, items in raw_by_strategy.items():
        qa_by_strategy[name] = attach_meta(items, name, rows_by_id)

    # Phase 5
    filtered, review_stats = phase_review(cfg, llm, qa_by_strategy, rows)

    # 合并 + 去重
    all_items: list[dict] = []
    for items in filtered.values():
        all_items.extend(items)
    print(f"[Merge] raw {sum(len(v) for v in filtered.values())} → 合并前 {len(all_items)}")
    all_items = dedup(all_items)
    print(f"[Merge] dedup 后 {len(all_items)}")

    # Phase 6: 写盘
    alpaca_path = f"{cfg.out_prefix}.alpaca.jsonl"
    chatml_path = f"{cfg.out_prefix}.chatml.jsonl"
    with open(alpaca_path, "w", encoding="utf-8") as f:
        for it in all_items:
            f.write(json.dumps(to_alpaca_record(it), ensure_ascii=False) + "\n")
    with open(chatml_path, "w", encoding="utf-8") as f:
        for it in all_items:
            f.write(json.dumps(to_chatml_record(it), ensure_ascii=False) + "\n")
    print(f"[Out] {len(all_items)} → {alpaca_path}")
    print(f"[Out] {len(all_items)} → {chatml_path}")

    stats = {
        "config": {k: v for k, v in asdict(cfg).items() if k != "api_key"},
        "rows": len(rows),
        "n_plans": len(plans),
        "n_codegen_success": len(success),
        "n_codegen_fail": len(codegen_results) - len(success),
        "qa_count": len(all_items),
        "by_strategy": {k: len(v) for k, v in filtered.items()},
        "by_qa_type": dict(Counter(it["qa_type"] for it in all_items)),
        "review_stats": review_stats,
        "codegen_results": [
            {"name": r["name"], "kind": r["kind"], "ok": r["ok"],
             "error": r.get("error", "")[:200]}
            for r in codegen_results
        ],
    }
    with open(f"{cfg.out_prefix}.stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[Out] stats → {cfg.out_prefix}.stats.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> Config:
    p = argparse.ArgumentParser(description="LLM-as-Coder MSDS QA 生成管线")
    p.add_argument("--input-json", default="抽取结果.json")
    p.add_argument("--out-prefix", default="qa_codegen")
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-base", type=float, default=2.0)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--cache-dir", default="cache_codegen")
    p.add_argument("--log-dir", default="codegen_logs")
    p.add_argument("--no-tools", action="store_true", help="禁用 OpenAI tools function calling")
    p.add_argument("--tool-loop-max", type=int, default=6)
    p.add_argument("--refine-max", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--sandbox-sample-rows", type=int, default=10)
    p.add_argument("--sandbox-timeout", type=int, default=30)
    p.add_argument("--sandbox-mem-mb", type=int, default=512)
    p.add_argument("--review-per-strategy", type=int, default=30)
    p.add_argument("--review-threshold", type=float, default=0.3,
                   help="抽样通过率 < 阈值 → 丢弃整策略；默认 0.3（仅干掉明显坏策略、保留中等质量策略的条级过滤结果）")
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260515)
    p.add_argument("--strategies-override", default="",
                   help="逗号分隔策略名，仅运行指定策略（在 plan 后过滤）")
    p.add_argument("--variants-per-slot", type=int, default=1,
                   help="模板渲染时每 (row, slot) 抽取多少条问句变体（默认 1；>1 会产生同答案多问句的训练对）")
    p.add_argument("--collapse-max-ratio", type=float, default=0.5,
                   help="若某策略输出中同一归一化问句占比 > 此阈值则丢弃整策略（默认 0.5）")
    a = p.parse_args()
    return Config(
        input_json=a.input_json, out_prefix=a.out_prefix, model=a.model,
        base_url=a.base_url, api_key=a.api_key,
        workers=a.workers, temperature=a.temperature, max_tokens=a.max_tokens,
        max_retries=a.max_retries, retry_base=a.retry_base, timeout=a.timeout,
        cache_dir=a.cache_dir, log_dir=a.log_dir,
        use_tools=not a.no_tools, tool_loop_max=a.tool_loop_max,
        refine_max=a.refine_max, batch_size=a.batch_size,
        sandbox_sample_rows=a.sandbox_sample_rows,
        sandbox_timeout=a.sandbox_timeout,
        sandbox_mem_mb=a.sandbox_mem_mb,
        review_per_strategy=a.review_per_strategy,
        review_threshold=a.review_threshold,
        max_rows=a.max_rows, seed=a.seed,
        strategies_override=a.strategies_override,
        variants_per_slot=a.variants_per_slot,
        collapse_max_ratio=a.collapse_max_ratio,
    )


if __name__ == "__main__":
    main(parse_args())
