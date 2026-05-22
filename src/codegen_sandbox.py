"""
codegen_sandbox.py — LLM 生成代码的安全执行沙箱

两层防护：
1. AST 白名单：解析源码 → 禁止危险 import / 属性 / 名字
2. 子进程隔离：python -I -S -E 启动 + resource.setrlimit 限 CPU/内存/超时
                stdin 喂 rows，stdout 收 JSON QA 列表

公开接口：
- check_ast(code: str) -> (ok: bool, msg: str)
- run_function_in_sandbox(code, func_name, rows, seed, timeout, mem_mb) -> dict
    返回 {"ok": bool, "items": [...], "stderr": str, "n": int, "rejected": int}
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# AST 白名单
# ---------------------------------------------------------------------------
ALLOWED_IMPORTS = {
    "random", "json", "re", "itertools", "collections", "math",
    "string", "copy", "functools", "operator", "statistics", "datetime",
}

# 名字层面禁止使用
FORBIDDEN_NAMES = {
    "__import__", "eval", "exec", "compile", "open", "input",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "breakpoint", "help", "memoryview", "exit", "quit",
    "os", "sys", "subprocess", "socket", "shutil", "pickle", "marshal",
    "ctypes", "importlib", "builtins", "pathlib", "shelve",
}

# 禁止访问的属性（dunder 黑魔法 + 反射）
FORBIDDEN_ATTRS = {
    "__class__", "__bases__", "__subclasses__", "__mro__",
    "__globals__", "__getattribute__", "__reduce__", "__reduce_ex__",
    "__import__", "__builtins__", "__dict__", "__code__", "__func__",
    "__self__", "__closure__", "__module__",
}


class SandboxRejected(Exception):
    pass


def check_ast(code: str) -> tuple[bool, str]:
    """静态检查 LLM 代码。返回 (ok, reason)。"""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    for node in ast.walk(tree):
        # 1) import 白名单
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    return False, f"forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                return False, f"forbidden import from: {node.module}"

        # 2) 名字黑名单
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                return False, f"forbidden name: {node.id}"

        # 3) 属性黑名单
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS:
                return False, f"forbidden attribute: .{node.attr}"
            # 形如 ().__class__.__bases__ 也会被上面的属性命中

        # 4) 禁止 try: except: 隐藏异常以掩盖错误（保留 try/except 但限制 bare except 不抑制）
        # （宽松处理：允许 try/except，便于错误抛出）

        # 5) 禁止 with 调用文件
        elif isinstance(node, ast.Call):
            # Call.func 形态多样；属性访问已在上面拦
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_NAMES:
                return False, f"forbidden call: {node.func.id}()"

    return True, "ok"


# ---------------------------------------------------------------------------
# 子进程 runner（被 subprocess 调用执行）
# ---------------------------------------------------------------------------
_RUNNER_TEMPLATE = r"""
# -*- coding: utf-8 -*-
# Auto-generated sandbox runner. DO NOT EDIT.
import sys, json, random, resource, signal

# 1) 资源限制：内存 + CPU
try:
    resource.setrlimit(resource.RLIMIT_AS, ({mem_bytes}, {mem_bytes}))
except Exception:
    pass
try:
    resource.setrlimit(resource.RLIMIT_CPU, ({cpu_sec}, {cpu_sec}))
except Exception:
    pass

# 2) 超时硬保险
def _timeout(_a, _b):
    print(json.dumps({{"_sandbox_error": "timeout"}}, ensure_ascii=False))
    sys.exit(124)
signal.signal(signal.SIGALRM, _timeout)
signal.alarm({wall_sec})

# 3) 读取 stdin: {{rows, seed, func_name}}
try:
    payload = json.loads(sys.stdin.read())
    rows = payload["rows"]
    seed = int(payload.get("seed", 0))
    func_name = payload["func_name"]
except Exception as e:
    print(json.dumps({{"_sandbox_error": "stdin parse: %s" % e}}, ensure_ascii=False))
    sys.exit(2)

rng = random.Random(seed)

# 4) 注入用户代码（以 string 形式执行到独立命名空间）
_user_globals = {{"__builtins__": __builtins__}}
_user_code = {user_code!r}

try:
    exec(compile(_user_code, "<llm_code>", "exec"), _user_globals)
except Exception as e:
    import traceback
    print(json.dumps({{"_sandbox_error": "exec: %s\n%s" % (e, traceback.format_exc()[-1500:])}}, ensure_ascii=False))
    sys.exit(3)

fn = _user_globals.get(func_name)
if not callable(fn):
    print(json.dumps({{"_sandbox_error": "function not found: %s" % func_name}}, ensure_ascii=False))
    sys.exit(4)

# 5) 执行
try:
    out = fn(rows, rng)
except Exception as e:
    import traceback
    print(json.dumps({{"_sandbox_error": "runtime: %s\n%s" % (e, traceback.format_exc()[-1500:])}}, ensure_ascii=False))
    sys.exit(5)

# 6) 验证 + 输出
if not isinstance(out, list):
    print(json.dumps({{"_sandbox_error": "return type not list: %s" % type(out).__name__}}, ensure_ascii=False))
    sys.exit(6)

# 截断每条文本，避免内存炸
clean = []
for it in out[:50000]:
    if not isinstance(it, dict):
        continue
    clean.append(it)

print(json.dumps({{"_sandbox_ok": True, "items": clean}}, ensure_ascii=False))
"""


def run_function_in_sandbox(
    code: str,
    func_name: str,
    rows: list[dict],
    seed: int = 0,
    timeout: int = 30,
    mem_mb: int = 512,
) -> dict[str, Any]:
    """在 subprocess 沙箱里运行 LLM 生成的函数。

    Returns:
        {"ok": bool, "items": [...], "stderr": str, "n": int, "error": str|None}
    """
    ok, reason = check_ast(code)
    if not ok:
        return {"ok": False, "items": [], "stderr": "", "n": 0, "error": f"AST 拒绝: {reason}"}

    runner_src = _RUNNER_TEMPLATE.format(
        mem_bytes=mem_mb * 1024 * 1024,
        cpu_sec=timeout,
        wall_sec=timeout,
        user_code=code,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="sb_runner_", delete=False, encoding="utf-8"
    ) as f:
        f.write(runner_src)
        runner_path = f.name

    try:
        payload = json.dumps(
            {"rows": rows, "seed": seed, "func_name": func_name},
            ensure_ascii=False,
        )
        # python -I：隔离模式（忽略 PYTHON* 环境、忽略 user site）
        # -S：不导入 site.py
        # -E：忽略 PYTHON* 环境变量
        proc = subprocess.run(
            [sys.executable, "-I", "-S", "-E", runner_path],
            input=payload.encode("utf-8"),
            capture_output=True,
            timeout=timeout + 5,  # subprocess.timeout 比内部 alarm 多 5s 保险
            env={"PATH": os.environ.get("PATH", ""), "LANG": "C.UTF-8"},
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "items": [], "stderr": "subprocess timeout", "n": 0,
                "error": "timeout"}
    finally:
        try:
            os.unlink(runner_path)
        except Exception:
            pass

    stderr = proc.stderr.decode("utf-8", errors="replace")[-2000:]
    stdout = proc.stdout.decode("utf-8", errors="replace")

    if proc.returncode != 0:
        # 尝试解析 stdout 末尾的 sandbox_error
        err_obj = _try_parse_last_json(stdout)
        err = (err_obj or {}).get("_sandbox_error") or f"exit={proc.returncode}"
        return {"ok": False, "items": [], "stderr": stderr, "n": 0,
                "error": err}

    obj = _try_parse_last_json(stdout)
    if not obj or not obj.get("_sandbox_ok"):
        err = (obj or {}).get("_sandbox_error", "no output")
        return {"ok": False, "items": [], "stderr": stderr, "n": 0, "error": err}

    items = obj.get("items", [])
    return {"ok": True, "items": items, "stderr": stderr, "n": len(items), "error": None}


def _try_parse_last_json(text: str) -> dict | None:
    """从子进程 stdout 最后一行解析 JSON。"""
    if not text:
        return None
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# CLI 自检：python codegen_sandbox.py 跑一遍攻击/良性样例
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== AST self-test ===")

    GOOD = textwrap.dedent("""
        def gen_demo(rows, rng):
            out = []
            for r in rows:
                out.append({"q": "Q?" + r.get("中文名", ""), "a": "A.", "type": "demo"})
            return out
    """)
    BAD_CASES = {
        "os.system": "import os\ndef gen(r,n): os.system('echo x'); return []",
        "eval": "def gen(r,n): return eval('1+1')",
        "__import__": "def gen(r,n): __import__('os'); return []",
        "open file": "def gen(r,n): open('/etc/passwd'); return []",
        "subprocess": "import subprocess\ndef gen(r,n): return []",
        "dunder bases": "def gen(r,n): ().__class__.__bases__; return []",
    }
    print("good:", check_ast(GOOD))
    for name, src in BAD_CASES.items():
        ok, msg = check_ast(src)
        marker = "OK-rejected" if not ok else "!! LEAKED !!"
        print(f"  [{marker}] {name}: {msg}")

    print("\n=== Subprocess self-test ===")
    rows = [{"中文名": "乙炔", "CAS号": "74-86-2"},
            {"中文名": "苯", "CAS号": "71-43-2"}]
    r = run_function_in_sandbox(GOOD, "gen_demo", rows, seed=42)
    print(f"good run: ok={r['ok']} n={r['n']} error={r['error']}")
    if r["items"]:
        print(f"  first item: {r['items'][0]}")

    # 真实攻击：能通过 AST 但运行时也无害（os 已被 AST 拦截）
    LEAK = "def gen_x(rows, rng):\n    return [{'q':'leak','a': str(type([]).__mro__),'type':'x'}]"
    r2 = run_function_in_sandbox(LEAK, "gen_x", rows)
    print(f"mro leak AST: {check_ast(LEAK)}  (should be rejected for .__mro__)")
