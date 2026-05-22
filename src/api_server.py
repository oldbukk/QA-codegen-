from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
RUNS_DIR = PROJECT_ROOT / "runs"
JOBS_DIR = RUNS_DIR / "jobs"
CACHE_DIR = RUNS_DIR / "cache_codegen"

TERMINAL_STATUSES = {"completed", "failed", "canceled"}

PHASE_UPDATES = [
    ("[Phase 1]", "inspect", 10, "计算输入 schema"),
    ("[Phase 2]", "plan", 25, "设计 QA 生成策略"),
    ("[Phase 3]", "codegen", 45, "生成并验证策略代码"),
    ("[Phase 4]", "full_run", 70, "批量生成 QA"),
    ("[Phase 5]", "review", 85, "抽样审核 QA"),
    ("[Merge]", "export", 92, "合并与去重"),
    ("[Out]", "export", 96, "写出结果"),
]


class JobConfig(BaseModel):
    out_prefix: str = "qa_codegen"
    model: str = Field(default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    base_url: str = Field(default=os.environ.get("OPENAI_BASE_URL", ""))
    api_key: str = Field(default=os.environ.get("OPENAI_API_KEY", ""))
    workers: int = 4
    temperature: float = 0.4
    max_tokens: int = 4096
    max_retries: int = 3
    retry_base: float = 2.0
    timeout: float = 120.0
    use_tools: bool = True
    tool_loop_max: int = 6
    refine_max: int = 3
    batch_size: int = 200
    sandbox_sample_rows: int = 10
    sandbox_timeout: int = 30
    sandbox_mem_mb: int = 512
    review_per_strategy: int = 30
    review_threshold: float = 0.3
    max_rows: int = 0
    seed: int = 20260515
    strategies_override: str = ""
    variants_per_slot: int = 1
    collapse_max_ratio: float = 0.5

    class Config:
        extra = "forbid"


class CreateJobRequest(BaseModel):
    records: list[dict[str, Any]]
    config: Optional[JobConfig] = None

    class Config:
        extra = "forbid"


app = FastAPI(title="QA Codegen API", version="2.0.0")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def model_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def read_job(job_id: str) -> dict[str, Any]:
    path = job_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="job not found")
    return json.loads(path.read_text(encoding="utf-8"))


def write_job(job: dict[str, Any]) -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job["updated_at"] = now()
    path = job_path(job["job_id"])
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def update_job(job_id: str, **changes: Any) -> dict[str, Any]:
    job = read_job(job_id)
    job.update(changes)
    write_job(job)
    return job


def make_job_id() -> str:
    return datetime.now().strftime("job_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]


def safe_out_prefix(value: str) -> str:
    value = (value or "qa_codegen").strip()
    if not value or value in {".", ".."} or Path(value).name != value:
        raise HTTPException(status_code=400, detail="config.out_prefix must be a plain file prefix")
    return value


def output_paths(run_dir: Path, out_prefix: str) -> dict[str, str]:
    return {
        "alpaca": str(run_dir / f"{out_prefix}.alpaca.jsonl"),
        "chatml": str(run_dir / f"{out_prefix}.chatml.jsonl"),
        "stats": str(run_dir / f"{out_prefix}.stats.json"),
        "log_dir": str(run_dir / "codegen_logs"),
    }


def refresh_latest(run_dir: Path) -> None:
    latest = RUNS_DIR / "latest"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(run_dir.resolve(), target_is_directory=True)
    except Exception:
        pass


def build_command(input_json: Path, config: dict[str, Any]) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(SRC_DIR / "build_qa_codegen.py"),
        "--input-json",
        str(input_json),
        "--out-prefix",
        config["out_prefix"],
        "--cache-dir",
        str(CACHE_DIR),
        "--log-dir",
        "codegen_logs",
        "--model",
        str(config["model"]),
        "--workers",
        str(config["workers"]),
        "--temperature",
        str(config["temperature"]),
        "--max-tokens",
        str(config["max_tokens"]),
        "--max-retries",
        str(config["max_retries"]),
        "--retry-base",
        str(config["retry_base"]),
        "--timeout",
        str(config["timeout"]),
        "--tool-loop-max",
        str(config["tool_loop_max"]),
        "--refine-max",
        str(config["refine_max"]),
        "--batch-size",
        str(config["batch_size"]),
        "--sandbox-sample-rows",
        str(config["sandbox_sample_rows"]),
        "--sandbox-timeout",
        str(config["sandbox_timeout"]),
        "--sandbox-mem-mb",
        str(config["sandbox_mem_mb"]),
        "--review-per-strategy",
        str(config["review_per_strategy"]),
        "--review-threshold",
        str(config["review_threshold"]),
        "--max-rows",
        str(config["max_rows"]),
        "--seed",
        str(config["seed"]),
        "--variants-per-slot",
        str(config["variants_per_slot"]),
        "--collapse-max-ratio",
        str(config["collapse_max_ratio"]),
    ]
    if config.get("base_url"):
        cmd.extend(["--base-url", str(config["base_url"])])
    if config.get("strategies_override"):
        cmd.extend(["--strategies-override", str(config["strategies_override"])])
    if not config.get("use_tools", True):
        cmd.append("--no-tools")
    return cmd


def apply_log_update(job_id: str, line: str) -> None:
    for marker, stage, progress, message in PHASE_UPDATES:
        if marker in line:
            job = read_job(job_id)
            if job.get("status") not in TERMINAL_STATUSES:
                job["status"] = "running"
                job["stage"] = stage
                job["progress"] = max(int(job.get("progress", 0)), progress)
                job["message"] = message
                write_job(job)
            return


def terminate_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)

    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"output file not found: {path.name}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_codegen_job(job_id: str, api_key: str, config: dict[str, Any]) -> None:
    job = read_job(job_id)
    if job.get("status") == "canceled":
        return

    run_dir = Path(job["run_dir"])
    input_json = Path(job["input_json"])
    cmd = build_command(input_json, config)
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key
    env["PYTHONPATH"] = str(SRC_DIR)
    if config.get("base_url"):
        env["OPENAI_BASE_URL"] = str(config["base_url"])

    update_job(
        job_id,
        status="running",
        stage="created",
        progress=1,
        message="任务启动中",
        started_at=now(),
    )

    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(run_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
            env=env,
        )
        update_job(job_id, pid=proc.pid, message="任务运行中")

        if proc.stdout:
            for line in proc.stdout:
                print(line, end="")
                apply_log_update(job_id, line)
                latest = read_job(job_id)
                if latest.get("status") == "canceled":
                    terminate_process_group(proc.pid)
                    break

        code = proc.wait()
        latest = read_job(job_id)
        if latest.get("status") == "canceled":
            update_job(job_id, pid=None, stage="canceled", progress=latest.get("progress", 0),
                       message="job canceled", ended_at=now(), returncode=code)
            return

        outputs = latest["outputs"]
        missing = [name for name, path in outputs.items()
                   if name != "log_dir" and not Path(path).exists()]
        if code != 0 or missing:
            error = f"codegen failed with exit code {code}"
            if missing:
                error += f"; missing outputs: {', '.join(missing)}"
            update_job(job_id, status="failed", stage="failed", progress=latest.get("progress", 0),
                       message="job failed", pid=None, error=error, ended_at=now(), returncode=code)
            return

        update_job(job_id, status="completed", stage="completed", progress=100,
                   message="done", pid=None, ended_at=now(), returncode=code)
    except Exception as exc:
        latest = read_job(job_id)
        if latest.get("status") == "canceled":
            update_job(job_id, pid=None, stage="canceled", message="job canceled", ended_at=now())
            return
        if proc and proc.poll() is None:
            terminate_process_group(proc.pid)
        update_job(job_id, status="failed", stage="failed", pid=None,
                   message="job failed", error=str(exc), ended_at=now())


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/jobs")
def create_job(req: CreateJobRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    if not req.records:
        raise HTTPException(status_code=400, detail="records must not be empty")

    cfg_model = req.config or JobConfig()
    config = model_dict(cfg_model)
    api_key = str(config.pop("api_key", "") or os.environ.get("OPENAI_API_KEY", ""))
    if not api_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY 未设置，且请求体 config 未提供 api_key")

    config["out_prefix"] = safe_out_prefix(str(config.get("out_prefix", "qa_codegen")))

    job_id = make_job_id()
    run_dir = RUNS_DIR / job_id
    run_dir.mkdir(parents=True, exist_ok=False)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    refresh_latest(run_dir)

    input_json = run_dir / "input.json"
    input_json.write_text(
        json.dumps({"records": req.records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    job = {
        "job_id": job_id,
        "status": "queued",
        "stage": "created",
        "progress": 0,
        "message": "",
        "pid": None,
        "run_dir": str(run_dir),
        "input_json": str(input_json),
        "outputs": output_paths(run_dir, config["out_prefix"]),
        "config": config,
        "error": None,
        "created_at": now(),
        "updated_at": now(),
        "started_at": None,
        "ended_at": None,
        "returncode": None,
    }
    write_job(job)

    background_tasks.add_task(run_codegen_job, job_id, api_key, config)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return read_job(job_id)


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    job = read_job(job_id)
    if job.get("status") in TERMINAL_STATUSES:
        return job

    pid = job.get("pid")
    job["status"] = "canceled"
    job["stage"] = "canceled"
    job["message"] = "job canceled"
    job["pid"] = None
    job["ended_at"] = now()
    write_job(job)

    if pid:
        terminate_process_group(int(pid))

    return read_job(job_id)


@app.get("/jobs/{job_id}/result")
def get_result(job_id: str) -> dict[str, Any]:
    job = read_job(job_id)
    if job.get("status") != "completed":
        raise HTTPException(status_code=409, detail=f"job is not completed: {job.get('status')}")

    outputs = job.get("outputs") or {}
    stats_path = Path(outputs.get("stats", ""))
    if not stats_path.exists():
        raise HTTPException(status_code=404, detail="stats output not found")

    return {
        "job_id": job_id,
        "status": job.get("status"),
        "stats": json.loads(stats_path.read_text(encoding="utf-8")),
        "alpaca": read_jsonl(Path(outputs.get("alpaca", ""))),
        "chatml": read_jsonl(Path(outputs.get("chatml", ""))),
        "outputs": outputs,
    }


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8001, reload=False)
