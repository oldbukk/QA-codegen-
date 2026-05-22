#!/usr/bin/env bash
# 一键运行入口：从方案根目录调用
# 用法：
#   ./run.sh                       # 全量
#   ./run.sh --max-rows 20         # 冒烟
# 前置：
#   export OPENAI_API_KEY=sk-xxx                            # 必需
#   export OPENAI_BASE_URL=https://api.openai-proxy.org/v1  # 可选
#
# 产物结构：
#   runs/
#     cache_codegen/                跨次共享的 LLM 调用缓存（按 hash）
#     run_<时间戳>/                 本次运行专属
#       plan.json schema.json
#       generated_generators/ generated_templates/
#       codegen_logs/
#       <out-prefix>.alpaca.jsonl   <out-prefix>.chatml.jsonl   <out-prefix>.stats.json
#     latest -> run_<最新时间戳>     软链，方便查看
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[run.sh] 错误：未设置 OPENAI_API_KEY 环境变量。" >&2
    echo "[run.sh] 请先执行：export OPENAI_API_KEY=sk-..." >&2
    exit 2
fi
RUNS_ROOT="$HERE/runs"
RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$RUNS_ROOT/$RUN_ID"
CACHE_DIR="$RUNS_ROOT/cache_codegen"
mkdir -p "$RUN_DIR" "$CACHE_DIR"
ln -sfn "$RUN_ID" "$RUNS_ROOT/latest"
cd "$RUN_DIR"
export PYTHONPATH="$HERE/src"
echo "[run.sh] 本次运行目录：$RUN_DIR"
echo "[run.sh] 共享缓存目录：$CACHE_DIR"
exec python3 "$HERE/src/build_qa_codegen.py" \
    --input-json "$HERE/data/抽取结果.json" \
    --cache-dir "$CACHE_DIR" \
    --log-dir "$RUN_DIR/codegen_logs" \
    "$@"
