import asyncio
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles


ROOT = Path(__file__).resolve().parent
MODEL_NAME = os.getenv("IMAGE_MODEL_NAME", "Qwen/Qwen-Image")
API_KEY = os.getenv("API_KEY", "local-dev-key")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))
WORKER_HOST = os.getenv("IMAGE_WORKER_HOST", "127.0.0.1")
WORKER_PORT = int(os.getenv("IMAGE_WORKER_PORT", "18001"))
IDLE_UNLOAD_SECONDS = int(os.getenv("IMAGE_IDLE_UNLOAD_SECONDS", "300"))
WORKER_START_TIMEOUT = int(os.getenv("IMAGE_WORKER_START_TIMEOUT", "120"))
REQUEST_TIMEOUT = float(os.getenv("IMAGE_REQUEST_TIMEOUT", "0")) or None
OUTPUT_DIR = Path(os.getenv("IMAGE_OUTPUT_DIR", ROOT / "outputs" / "qwen-image"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WORKER_URL = f"http://{WORKER_HOST}:{WORKER_PORT}"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Qwen-Image lazy proxy", version="0.1.0")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

_worker: Optional[subprocess.Popen] = None
_worker_log = None
_last_used = 0.0
_active_requests = 0
_worker_lock = asyncio.Lock()


def _touch() -> None:
    global _last_used
    _last_used = time.monotonic()


def _worker_running() -> bool:
    return _worker is not None and _worker.poll() is None


def _worker_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "HOST": WORKER_HOST,
            "PORT": str(WORKER_PORT),
            "IMAGE_MODEL_NAME": MODEL_NAME,
            "IMAGE_MODEL_PATH": env.get("IMAGE_MODEL_PATH", str(ROOT / "models" / "Qwen-Image")),
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def _check_api_key(request: Request) -> None:
    authorization = request.headers.get("authorization", "")
    expected = f"Bearer {API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


async def _wait_worker_ready() -> None:
    deadline = time.monotonic() + WORKER_START_TIMEOUT
    timeout = httpx.Timeout(2.0, connect=2.0)
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        while time.monotonic() < deadline:
            if not _worker_running():
                raise RuntimeError("Qwen-Image worker exited during startup. Check logs/qwen-image-worker.log")
            try:
                response = await client.get(f"{WORKER_URL}/health")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
    raise RuntimeError("Qwen-Image worker did not become ready in time")


async def _ensure_worker() -> None:
    global _worker, _worker_log
    if _worker_running():
        return
    async with _worker_lock:
        if _worker_running():
            return
        if _worker_log is not None:
            _worker_log.close()
        _worker_log = open(LOG_DIR / "qwen-image-worker.log", "ab", buffering=0)
        _worker = subprocess.Popen(
            [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "qwen_image_worker.py")],
            cwd=str(ROOT),
            env=_worker_env(),
            stdout=_worker_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        await _wait_worker_ready()


def _terminate_worker() -> None:
    global _worker, _worker_log
    if _worker is None:
        return
    if _worker.poll() is None:
        try:
            os.killpg(_worker.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            _worker.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(_worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _worker.wait(timeout=10)
    _worker = None
    if _worker_log is not None:
        _worker_log.close()
        _worker_log = None


async def _idle_reaper() -> None:
    while True:
        await asyncio.sleep(15)
        if not _worker_running() or _active_requests:
            continue
        if time.monotonic() - _last_used >= IDLE_UNLOAD_SECONDS:
            _terminate_worker()


@app.on_event("startup")
async def startup_event() -> None:
    _touch()
    asyncio.create_task(_idle_reaper())


@app.on_event("shutdown")
async def shutdown_event() -> None:
    _terminate_worker()


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "model": MODEL_NAME,
        "worker_running": _worker_running(),
        "idle_unload_seconds": IDLE_UNLOAD_SECONDS,
        "worker_url": WORKER_URL,
    }


async def _forward_json(path: str, payload: Dict[str, Any]) -> JSONResponse:
    global _active_requests
    await _ensure_worker()
    _active_requests += 1
    _touch()
    timeout = None if REQUEST_TIMEOUT is None else httpx.Timeout(REQUEST_TIMEOUT)
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.post(f"{WORKER_URL}{path}", json=payload)
        try:
            content = response.json()
        except ValueError:
            content = {"detail": response.text}
        return JSONResponse(status_code=response.status_code, content=content)
    finally:
        _active_requests -= 1
        _touch()


@app.post("/v1/images/generations")
async def image_generations(request: Request) -> Any:
    _check_api_key(request)
    payload = await request.json()
    return await _forward_json("/v1/images/generations", payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("proxy_qwen_image:app", host=HOST, port=PORT, reload=False)
