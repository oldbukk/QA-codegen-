# QA-codegen 方案 V2

这是一个把化学品 MSDS 结构化字段转换成 QA 训练数据的工具。V2 版本使用 **JSON 作为输入**，并提供 **异步 API**：创建任务后立即返回 `job_id`，前端或调用方可以轮询状态、取消任务、任务完成后再获取结果。

本方案不是让 LLM 逐行直接写 QA，而是让 LLM 先像代码工程师一样工作：读取字段 schema，规划 QA 策略，生成 Python 代码或模板，本地沙箱执行，再抽样审核，最后输出 Alpaca 和 ChatML 两种格式。

## 你能得到什么

运行完成后会得到 3 类主要文件：

```text
<out-prefix>.alpaca.jsonl   # Alpaca / instruction 格式
<out-prefix>.chatml.jsonl   # ChatML messages 格式
<out-prefix>.stats.json     # 统计信息、策略通过情况、配置记录
```

API 模式下也可以直接通过 `GET /jobs/{job_id}/result` 得到结构化 JSON，里面包含 `stats`、`alpaca`、`chatml` 和输出文件路径。

---

## 一、先确认你要用哪种方式

推荐两种使用方式：

| 方式 | 适合谁 | 怎么用 |
|---|---|---|
| API 异步任务 | 前端、服务集成、需要轮询/取消任务 | 启动 `src/api_server.py`，调用 `/jobs` |
| CLI 命令行 | 本地一次性生成数据、调试参数 | 直接运行 `./run.sh` |

如果你只是想最快验证流程，先用 CLI；如果你要接前端或其他系统，用 API。

---

## 二、准备环境

进入 V2 目录：

```bash
cd /home/ubuntu/下载/QA-extrect/QA-codegen方案V2/QA-codegen方案
```

安装依赖：

```bash
pip install -r requirements.txt
```

设置模型 API：

```bash
export OPENAI_API_KEY=sk-你的key
export OPENAI_BASE_URL=https://api.openai-proxy.org/v1
```

说明：

- `OPENAI_API_KEY` 必须设置，除非你在 API 请求体的 `config.api_key` 里传。
- `OPENAI_BASE_URL` 可选。如果你使用官方 OpenAI 接口，可以不设置；如果使用兼容代理，通常要带 `/v1` 后缀。
- 不建议把真实 key 写进 README、脚本或请求文件。API 服务会把 key 放进子进程环境变量，不会写入 job 文件或命令行参数。

---

## 三、输入 JSON 格式

V2 不再以 CSV 作为运行输入。默认样例文件是：

```text
data/抽取结果.json
```

支持两种 JSON 形态。

第一种，推荐使用对象包一层 `records`：

```json
{
  "records": [
    {
      "中文名": "乙炔",
      "英文名": "acetylene",
      "CAS号": "74－86－2",
      "危险特性": "极易燃烧爆炸。",
      "吸入": "迅速脱离现场至空气新鲜处。保持呼吸道通畅。"
    }
  ]
}
```

第二种，也支持裸数组：

```json
[
  {
    "中文名": "乙炔",
    "英文名": "acetylene",
    "CAS号": "74－86－2"
  }
]
```

字段要求：

- 每条 record 必须是一个 JSON object。
- 建议至少包含 `中文名` 或其他名称字段，否则该行可能会被跳过。
- 字段名可以是中文，例如 `危险特性`、`消防措施`、`吸入`、`泄漏处理`。
- 空字符串、`无`、`无资料`、`未制定标准`、`暂无` 等会被当作空值，不生成 QA。
- 程序会自动给每行补内部字段 `_主名`、`_别名`、`_行号`，用于生成和追踪 QA。

---

## 四、最简单的 CLI 用法

### 1. 冒烟测试

先只跑一小部分，确认环境、key、模型接口都可用：

```bash
./run.sh --max-rows 20 --review-per-strategy 5 --out-prefix qa_smoke
```

### 2. 查看输出

```bash
ls runs/latest/
```

你会看到类似：

```text
qa_smoke.alpaca.jsonl
qa_smoke.chatml.jsonl
qa_smoke.stats.json
schema.json
plan.json
codegen_logs/
generated_generators/
generated_templates/
```

### 3. 全量运行

确认小样本没问题后再跑全量：

```bash
./run.sh --workers 4 --review-per-strategy 30 --out-prefix qa_codegen
```

### 4. 换自己的输入文件

```bash
python src/build_qa_codegen.py \
  --input-json /path/to/your_input.json \
  --out-prefix my_qa \
  --cache-dir runs/cache_codegen \
  --log-dir runs/manual_run_logs \
  --max-rows 20 \
  --review-per-strategy 5
```

常用建议：

- 第一次跑：加 `--max-rows 20 --review-per-strategy 5`。
- 正式跑：去掉 `--max-rows`，把 `--review-per-strategy` 调到 `30`。
- 模型慢或接口容易超时：先把 `--workers` 设成 `1` 或 `2`。

---

## 五、最简单的 API 用法

API 适合前端或业务系统调用。它不会阻塞等待整条管线完成，而是立即返回任务号。

### 1. 启动服务

```bash
python src/api_server.py
```

默认监听：

```text
http://0.0.0.0:8001
```

### 2. 检查服务是否启动

新开一个终端：

```bash
curl http://127.0.0.1:8001/health
```

正常返回：

```json
{"ok": true}
```

### 3. 准备请求文件

创建一个最小请求文件：

```bash
cat > /tmp/qa_job.json <<'JSON'
{
  "records": [
    {
      "中文名": "乙炔",
      "英文名": "acetylene",
      "CAS号": "74－86－2",
      "危险特性": "极易燃烧爆炸。",
      "吸入": "迅速脱离现场至空气新鲜处。保持呼吸道通畅。"
    }
  ],
  "config": {
    "out_prefix": "qa_api_smoke",
    "model": "gpt-4.1-mini",
    "max_rows": 1,
    "review_per_strategy": 1,
    "workers": 1
  }
}
JSON
```

也可以直接使用完整样例数据：

```bash
python - <<'PY'
import json
from pathlib import Path

records = json.loads(Path("data/抽取结果.json").read_text(encoding="utf-8"))["records"]
body = {
    "records": records,
    "config": {
        "out_prefix": "qa_api_full",
        "model": "gpt-4.1-mini",
        "workers": 4,
        "review_per_strategy": 30
    }
}
Path("/tmp/qa_job.json").write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
PY
```

### 4. 创建任务

```bash
curl -X POST http://127.0.0.1:8001/jobs \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/qa_job.json
```

正常返回：

```json
{"job_id":"job_20260522_224228_87f13a51","status":"queued"}
```

记住返回的 `job_id`，后面查询、取消、取结果都要用它。

### 5. 查询任务状态

把下面命令里的 `{job_id}` 换成真实任务号：

```bash
curl http://127.0.0.1:8001/jobs/{job_id}
```

返回示例：

```json
{
  "job_id": "job_20260522_224228_87f13a51",
  "status": "running",
  "stage": "codegen",
  "progress": 45,
  "message": "生成并验证策略代码",
  "error": null
}
```

状态说明：

| status | 含义 |
|---|---|
| `queued` | 已创建，等待后台执行 |
| `running` | 正在运行 |
| `completed` | 已完成，可以取结果 |
| `failed` | 失败，查看 `error` |
| `canceled` | 已取消 |

阶段说明：

| stage | 含义 |
|---|---|
| `created` | 任务刚创建 |
| `inspect` | 正在读取 JSON 并计算 schema |
| `plan` | LLM 正在规划 QA 策略 |
| `codegen` | LLM 正在生成策略代码或模板 |
| `full_run` | 正在批量执行生成 QA |
| `review` | 正在 LLM 抽样审核 |
| `export` | 正在合并、去重、写出结果 |
| `completed` | 已完成 |
| `failed` | 已失败 |
| `canceled` | 已取消 |

### 6. 取消任务

```bash
curl -X POST http://127.0.0.1:8001/jobs/{job_id}/cancel
```

取消会终止主管线子进程组。已经完成、失败或取消的任务，再次取消会直接返回当前状态。

### 7. 获取结果

只有 `status=completed` 后才能取结果：

```bash
curl http://127.0.0.1:8001/jobs/{job_id}/result
```

返回结构：

```json
{
  "job_id": "job_20260522_224228_87f13a51",
  "status": "completed",
  "stats": {},
  "alpaca": [],
  "chatml": [],
  "outputs": {
    "alpaca": "runs/<job_id>/<out-prefix>.alpaca.jsonl",
    "chatml": "runs/<job_id>/<out-prefix>.chatml.jsonl",
    "stats": "runs/<job_id>/<out-prefix>.stats.json",
    "log_dir": "runs/<job_id>/codegen_logs"
  }
}
```

如果任务没完成就取结果，会返回 `409`：

```json
{"detail":"job is not completed: running"}
```

---

## 六、API 请求参数怎么填

`POST /jobs` 顶层只有两个字段：

| 字段 | 必填 | 说明 |
|---|---|---|
| `records` | 是 | JSON records 列表，每个元素是一行化学品字段 |
| `config` | 否 | 运行参数，不填则使用默认值 |

常用 `config`：

| 参数 | 默认值 | 建议 |
|---|---|---|
| `out_prefix` | `qa_codegen` | 输出文件名前缀，例如 `qa_smoke` |
| `model` | `gpt-4.1-mini` | 模型名，兼容模型直接替换这里 |
| `base_url` | 环境变量 `OPENAI_BASE_URL` | 使用代理时填 `https://.../v1` |
| `api_key` | 环境变量 `OPENAI_API_KEY` | 不建议写请求文件，优先用环境变量 |
| `workers` | `4` | 并发策略数；接口不稳定时设为 `1` |
| `max_rows` | `0` | `0` 表示全量；测试时建议 `1`、`5`、`20` |
| `review_per_strategy` | `30` | 测试设 `1` 或 `5`，正式设 `30` |
| `strategies_override` | 空 | 只跑指定策略，例如 `scenario_inhale` |
| `sandbox_sample_rows` | `10` | codegen 阶段沙箱验证样本数 |
| `batch_size` | `200` | full run 阶段每批处理行数 |

最小可用配置：

```json
{
  "records": [{ "中文名": "乙炔", "吸入": "迅速脱离现场至空气新鲜处。" }],
  "config": {
    "out_prefix": "qa_test",
    "max_rows": 1,
    "review_per_strategy": 1
  }
}
```

只跑一个策略，适合快速端到端验证：

```json
{
  "records": [{ "中文名": "乙炔", "吸入": "迅速脱离现场至空气新鲜处。" }],
  "config": {
    "out_prefix": "qa_e2e",
    "max_rows": 1,
    "review_per_strategy": 1,
    "workers": 1,
    "strategies_override": "scenario_inhale"
  }
}
```

---

## 七、输出文件怎么看

一次运行会生成一个独立目录。

CLI 模式：

```text
runs/run_<YYYYMMDD_HHMMSS>/
```

API 模式：

```text
runs/<job_id>/
```

目录里常见文件：

| 文件或目录 | 说明 |
|---|---|
| `input.json` | API 模式保存的本次输入 |
| `schema.json` | 从输入 records 计算出的字段 schema |
| `plan.json` | LLM 规划出的 QA 生成策略 |
| `generated_generators/` | LLM 生成的 Python 函数策略 |
| `generated_templates/` | LLM 生成的模板策略 |
| `codegen_logs/` | 每个策略的 LLM 对话和工具调用日志 |
| `<prefix>.alpaca.jsonl` | Alpaca 格式训练数据 |
| `<prefix>.chatml.jsonl` | ChatML 格式训练数据 |
| `<prefix>.stats.json` | 行数、策略数、QA 数、review 通过率等统计 |

查看最近一次 CLI 输出：

```bash
ls runs/latest/
```

查看生成了多少条 QA：

```bash
python - <<'PY'
import json
from pathlib import Path

stats = json.loads(Path("runs/latest/qa_codegen.stats.json").read_text(encoding="utf-8"))
print("QA count:", stats["qa_count"])
print("By type:", stats["by_qa_type"])
PY
```

如果你的 `out_prefix` 不是 `qa_codegen`，把上面的文件名换成实际前缀。

---

## 八、这个系统内部怎么工作

整体流程如下：

```text
1. Inspect
   读取 JSON records，清洗字段，识别主名、别名、行号，生成 schema.json

2. Plan
   LLM 根据 schema 规划多个 QA 生成策略，写入 plan.json

3. Codegen
   LLM 为每个策略生成模板 JSON 或 Python 函数

4. Sandbox
   Python 函数会先经过 AST 白名单检查，再进入独立 subprocess 沙箱执行

5. Full Run
   对全部输入 records 批量执行通过验证的策略

6. Review
   LLM 对每个策略抽样审核，质量过低的策略会被丢弃

7. Dedup
   合并所有策略结果，做严格去重和语义去重

8. Export
   输出 Alpaca JSONL、ChatML JSONL、stats JSON
```

为什么这样设计：

- LLM 调用次数比逐行生成少得多。
- 生成代码和模板会被保存，便于审计和复用。
- 沙箱限制 LLM 生成代码的行为，降低执行风险。
- Review 和去重能减少低质量、重复或自指的 QA。

---

## 九、目录结构

```text
QA-codegen方案/
├── README.md
├── run.sh
├── requirements.txt
├── data/
│   └── 抽取结果.json
├── src/
│   ├── api_server.py           # FastAPI 异步任务接口
│   ├── build_qa_codegen.py     # 主管线和 CLI
│   ├── codegen_prompts.py      # Prompt 模板
│   ├── codegen_tools.py        # LLM tool calling 数据探查工具
│   ├── codegen_sandbox.py      # AST 检查和 subprocess 沙箱
│   └── codegen_runner.py       # 模板渲染、QA 校验、去重工具
├── artifacts/                  # 参考中间产物
├── sample_outputs/             # 样例输出
└── runs/                       # 运行产物，自动生成，默认不提交 git
```

---

## 十、常见问题

### 1. 创建任务时报 `OPENAI_API_KEY 未设置`

先设置环境变量：

```bash
export OPENAI_API_KEY=sk-你的key
```

然后重新启动 API 服务：

```bash
python src/api_server.py
```

### 2. 代理接口报错或连不上

确认 `OPENAI_BASE_URL` 带 `/v1`：

```bash
export OPENAI_BASE_URL=https://api.openai-proxy.org/v1
```

也可以在 API 请求的 `config.base_url` 里传。

### 3. 任务一直在 `plan` 或 `codegen`

这是正常现象。`plan` 和 `codegen` 阶段会真实调用 LLM，尤其是第一次没有缓存时会较慢。

想快速验证可以这样配置：

```json
{
  "config": {
    "max_rows": 1,
    "review_per_strategy": 1,
    "workers": 1,
    "strategies_override": "scenario_inhale"
  }
}
```

### 4. 任务失败，怎么排查

先看任务状态：

```bash
curl http://127.0.0.1:8001/jobs/{job_id}
```

再看运行目录：

```bash
ls runs/{job_id}/
ls runs/{job_id}/codegen_logs/
```

重点查看：

- `error` 字段
- `codegen_logs/*.json`
- `<out-prefix>.stats.json`

### 5. 为什么输出数量比预期少

常见原因：

- 输入字段很多是空值或占位符。
- LLM review 丢弃了低质量策略或条目。
- 去重阶段合并了重复问法。
- 只设置了 `max_rows`，没有跑全量。
- 使用了 `strategies_override`，只跑了部分策略。

### 6. 如何减少费用和时间

测试阶段：

```json
{
  "config": {
    "max_rows": 5,
    "review_per_strategy": 1,
    "workers": 1
  }
}
```

正式阶段：

```json
{
  "config": {
    "max_rows": 0,
    "review_per_strategy": 30,
    "workers": 4
  }
}
```

如果同样的 prompt 和配置命中缓存，会复用 `runs/cache_codegen/`，不会重复调用 LLM。

---

## 十一、开发检查命令

语法检查：

```bash
python -m py_compile src/*.py
```

JSON 加载检查：

```bash
PYTHONPATH=src python - <<'PY'
from build_qa_codegen import load_rows, build_schema

rows = load_rows("data/抽取结果.json")
schema = build_schema(rows, sample_n=2, seed=1)
print("rows:", len(rows))
print("columns:", len(schema["columns"]))
print("first name:", rows[0].get("_主名"))
PY
```

沙箱自检：

```bash
python src/codegen_sandbox.py
```

---

