import asyncio
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3.6-27B")
API_KEY = os.getenv("API_KEY", "sk-123456789")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
WORKER_HOST = os.getenv("WORKER_HOST", "127.0.0.1")
WORKER_PORT = int(os.getenv("WORKER_PORT", "18000"))
IDLE_UNLOAD_SECONDS = int(os.getenv("IDLE_UNLOAD_SECONDS", "300"))
WORKER_START_TIMEOUT = int(os.getenv("WORKER_START_TIMEOUT", "120"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "0")) or None

WORKER_URL = f"http://{WORKER_HOST}:{WORKER_PORT}"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Qwen3.6-27B lazy proxy", version="0.2.0")

_worker: Optional[subprocess.Popen] = None
_worker_log = None
_last_used = 0.0
_active_requests = 0
_worker_lock = asyncio.Lock()


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    extra_body: Dict[str, Any] = Field(default_factory=dict)


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
            "MODEL_NAME": MODEL_NAME,
            "MODEL_PATH": env.get("MODEL_PATH", str(ROOT / "models" / "Qwen3.6-27B")),
            "IDLE_UNLOAD_SECONDS": str(max(IDLE_UNLOAD_SECONDS * 24, 3600)),
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
                raise HTTPException(status_code=503, detail="Model worker exited during startup. Check logs/worker.log.")
            try:
                response = await client.get(f"{WORKER_URL}/health")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
    raise HTTPException(status_code=503, detail="Model worker did not become ready in time.")


async def _ensure_worker() -> None:
    global _worker, _worker_log
    if _worker_running():
        return

    async with _worker_lock:
        if _worker_running():
            return
        if _worker_log is not None:
            _worker_log.close()
        _worker_log = open(LOG_DIR / "worker.log", "ab", buffering=0)
        _worker = subprocess.Popen(
            [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "serve_qwen36.py")],
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


@app.get("/v1/models")
async def list_models(request: Request) -> Dict[str, Any]:
    _check_api_key(request)
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "created": 0, "owned_by": "local"}],
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
        return JSONResponse(status_code=response.status_code, content=response.json())
    finally:
        _active_requests -= 1
        _touch()


async def _forward_stream(path: str, payload: Dict[str, Any]) -> AsyncIterator[bytes]:
    global _active_requests
    await _ensure_worker()
    _active_requests += 1
    _touch()
    timeout = None if REQUEST_TIMEOUT is None else httpx.Timeout(REQUEST_TIMEOUT)
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            async with client.stream("POST", f"{WORKER_URL}{path}", json=payload) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    yield body
                    return
                async for chunk in response.aiter_bytes():
                    yield chunk
    finally:
        _active_requests -= 1
        _touch()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    _check_api_key(request)
    payload = await request.json()
    if payload.get("stream"):
        return StreamingResponse(_forward_stream("/v1/chat/completions", payload), media_type="text/event-stream")
    return await _forward_json("/v1/chat/completions", payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("proxy_qwen36:app", host=HOST, port=PORT, reload=False)
