import asyncio
import os
import socket
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
MODEL_REGISTRY_PATH = Path(os.getenv("MODEL_REGISTRY_PATH", ROOT / "model_registry.json"))
HOST = os.getenv("WEB_HOST", "0.0.0.0")
PORT = int(os.getenv("WEB_PORT", "8080"))
DEFAULT_API_KEY = os.getenv("API_KEY", "sk-123456789")
REQUEST_TIMEOUT = float(os.getenv("WEB_REQUEST_TIMEOUT", "0")) or None
GATEWAY_NAME = os.getenv("GATEWAY_NAME", "local-ai-stack")
LAN_HOST = os.getenv("LAN_HOST", "")
STACK_PROBE_TIMEOUT = float(os.getenv("STACK_PROBE_TIMEOUT", "1.5"))
GATEWAY_AUTH = os.getenv("WEB_GATEWAY_AUTH", "1").lower() not in {"0", "false", "no"}
CORS_ORIGINS = [item.strip() for item in os.getenv("WEB_CORS_ORIGINS", "*").split(",") if item.strip()]

app = FastAPI(title="Local model web UI", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    enable_thinking: Optional[bool] = None
    extra_body: Dict[str, Any] = Field(default_factory=dict)


class ImageRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    negative_prompt: Optional[str] = " "
    size: Optional[str] = None
    steps: Optional[int] = None
    true_cfg_scale: Optional[float] = None
    seed: Optional[int] = None
    n: int = 1
    add_magic_prompt: bool = False
    extra_body: Dict[str, Any] = Field(default_factory=dict)


def _load_registry() -> Dict[str, Any]:
    if not MODEL_REGISTRY_PATH.exists():
        raise HTTPException(status_code=500, detail=f"Model registry not found: {MODEL_REGISTRY_PATH}")
    try:
        import json

        registry = json.loads(MODEL_REGISTRY_PATH.read_text())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read model registry: {exc}") from exc
    if not isinstance(registry.get("models"), list):
        raise HTTPException(status_code=500, detail="Model registry must contain a models list")
    return registry


def _public_model(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": entry.get("id"),
        "label": entry.get("label") or entry.get("id"),
        "type": entry.get("type"),
        "runner": entry.get("runner") or "openai-compatible",
        "base_url": entry.get("base_url"),
        "default": bool(entry.get("default")),
        "defaults": entry.get("defaults") or {},
        "sizes": entry.get("sizes") or [],
        "features": entry.get("features") or [],
    }


def _models(kind: Optional[str] = None) -> List[Dict[str, Any]]:
    items = _load_registry()["models"]
    if kind is not None:
        items = [item for item in items if item.get("type") == kind]
    return items


def _find_model(kind: str, model_id: Optional[str]) -> Dict[str, Any]:
    candidates = _models(kind)
    if not candidates:
        raise HTTPException(status_code=500, detail=f"No {kind} models are configured")
    if model_id:
        for item in candidates:
            if item.get("id") == model_id:
                return item
        raise HTTPException(status_code=400, detail=f"Unknown {kind} model: {model_id}")
    for item in candidates:
        if item.get("default"):
            return item
    return candidates[0]


def _api_key(model: Dict[str, Any]) -> str:
    env_name = str(model.get("api_key_env") or "API_KEY")
    return str(model.get("api_key") or os.getenv(env_name, DEFAULT_API_KEY))


def _endpoint(model: Dict[str, Any], suffix: str) -> str:
    base_url = str(model.get("base_url") or "").rstrip("/")
    if not base_url:
        raise HTTPException(status_code=500, detail=f"Model {model.get('id')} has no base_url")
    return f"{base_url}/{suffix.lstrip('/')}"


def _http_timeout() -> httpx.Timeout:
    if REQUEST_TIMEOUT is None:
        return httpx.Timeout(None)
    return httpx.Timeout(REQUEST_TIMEOUT, connect=10.0)


def _short_timeout() -> httpx.Timeout:
    return httpx.Timeout(STACK_PROBE_TIMEOUT, connect=STACK_PROBE_TIMEOUT)


def _check_gateway_key(request: Request) -> None:
    if not GATEWAY_AUTH:
        return
    expected = f"Bearer {DEFAULT_API_KEY}"
    if request.headers.get("authorization", "") != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _detect_lan_host() -> str:
    if LAN_HOST:
        return LAN_HOST
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("1.1.1.1", 80))
        host = sock.getsockname()[0]
    except OSError:
        host = "127.0.0.1"
    finally:
        sock.close()
    return host


def _public_base_url() -> str:
    return os.getenv("PUBLIC_BASE_URL", f"http://{_detect_lan_host()}:{PORT}").rstrip("/")


def _health_url(model: Dict[str, Any]) -> str:
    if model.get("health_url"):
        return str(model["health_url"])
    base_url = str(model.get("base_url") or "").rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    return f"{base_url}/health"


async def _probe_model(model: Dict[str, Any]) -> Dict[str, Any]:
    url = _health_url(model)
    result: Dict[str, Any] = {
        "id": model.get("id"),
        "label": model.get("label") or model.get("id"),
        "type": model.get("type"),
        "runner": model.get("runner") or "openai-compatible",
        "base_url": model.get("base_url"),
        "health_url": url,
        "status": "offline",
        "detail": None,
    }
    try:
        async with httpx.AsyncClient(timeout=_short_timeout(), trust_env=False) as client:
            response = await client.get(url)
    except httpx.HTTPError as exc:
        result["detail"] = str(exc)
        return result

    result["http_status"] = response.status_code
    if response.status_code >= 400:
        result["detail"] = response.text[:300]
        return result

    result["status"] = "ready"
    try:
        data = response.json()
    except ValueError:
        data = {}
    result["worker_running"] = data.get("worker_running")
    result["loaded"] = data.get("loaded")
    result["idle_unload_seconds"] = data.get("idle_unload_seconds")
    result["model"] = data.get("model") or model.get("id")
    return result


def _layer_status(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "ready"
    return "ready" if any(item.get("status") == "ready" for item in items) else "offline"


def _stack_layers(health: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chat = [item for item in health if item.get("type") == "chat"]
    image = [item for item in health if item.get("type") == "image"]
    return [
        {
            "name": "UI",
            "status": "ready",
            "primary": "Local console",
            "detail": "Open WebUI-compatible endpoint is exposed by this process.",
        },
        {
            "name": "Gateway",
            "status": "ready",
            "primary": f"{GATEWAY_NAME} /v1",
            "detail": "Routes chat and image requests to registered OpenAI-compatible backends.",
        },
        {
            "name": "LLM Serving",
            "status": _layer_status(chat),
            "primary": "Lazy Qwen worker",
            "detail": "Current local runner, with vLLM/SGLang kept as the high-throughput target layer.",
        },
        {
            "name": "Image Workflow",
            "status": _layer_status(image),
            "primary": "Qwen-Image worker",
            "detail": "Current local image runner, with ComfyUI/LocalAI as workflow and multi-modal expansion layers.",
        },
        {
            "name": "Operations",
            "status": "ready",
            "primary": "logs/ + outputs/",
            "detail": "Local logs, generated images, and reference snapshots stay inside this project.",
        },
    ]


def _integration_snippets(public_base_url: str) -> Dict[str, str]:
    gateway_url = f"{public_base_url}/v1"
    chat_payload = '{"model":"Qwen/Qwen3.6-27B","messages":[{"role":"user","content":"Say OK"}],"max_tokens":8}'
    image_payload = '{"model":"Qwen/Qwen-Image-2512","prompt":"A red circle on white background","size":"512x512","n":1}'
    return {
        "open_webui": "\n".join(
            [
                f"OpenAI API Base URL: {gateway_url}",
                f"OpenAI API Key: {DEFAULT_API_KEY}",
                "Model IDs: Qwen/Qwen3.6-27B, Qwen/Qwen-Image-2512",
            ]
        ),
        "litellm_yaml": "\n".join(
            [
                "model_list:",
                "  - model_name: qwen-chat",
                "    litellm_params:",
                "      model: openai/Qwen/Qwen3.6-27B",
                f"      api_base: {gateway_url}",
                f"      api_key: {DEFAULT_API_KEY}",
                "  - model_name: qwen-image",
                "    litellm_params:",
                "      model: openai/Qwen/Qwen-Image-2512",
                f"      api_base: {gateway_url}",
                f"      api_key: {DEFAULT_API_KEY}",
            ]
        ),
        "chat_curl": f"curl {gateway_url}/chat/completions -H 'Content-Type: application/json' -H 'Authorization: Bearer {DEFAULT_API_KEY}' -d '{chat_payload}'",
        "image_curl": f"curl {gateway_url}/images/generations -H 'Content-Type: application/json' -H 'Authorization: Bearer {DEFAULT_API_KEY}' -d '{image_payload}'",
    }


async def _post_json(model: Dict[str, Any], suffix: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {_api_key(model)}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=_http_timeout(), trust_env=False) as client:
            response = await client.post(_endpoint(model, suffix), json=payload, headers=headers)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream API is not reachable: {exc}") from exc

    if response.status_code >= 400:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise HTTPException(status_code=response.status_code, detail=detail)

    try:
        return response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Upstream API returned invalid JSON") from exc


async def _stream_upstream(model: Dict[str, Any], suffix: str, payload: Dict[str, Any]):
    headers = {"Authorization": f"Bearer {_api_key(model)}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=_http_timeout(), trust_env=False) as client:
            async with client.stream("POST", _endpoint(model, suffix), json=payload, headers=headers) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    yield body
                    return
                async for chunk in response.aiter_bytes():
                    yield chunk
    except httpx.HTTPError as exc:
        yield str({"detail": f"Upstream API is not reachable: {exc}"}).encode("utf-8")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    registry = _load_registry()
    return {"ok": True, "models": len(registry["models"]), "gateway": f"{_public_base_url()}/v1"}


@app.get("/api/models")
async def list_models() -> Dict[str, Any]:
    items = [_public_model(item) for item in _models()]
    return {
        "models": items,
        "chat": [item for item in items if item.get("type") == "chat"],
        "image": [item for item in items if item.get("type") == "image"],
    }


@app.get("/api/stack")
async def stack() -> Dict[str, Any]:
    registry = _load_registry()
    models = registry["models"]
    health_items = await asyncio.gather(*[_probe_model(item) for item in models])
    public_base = _public_base_url()
    return {
        "ok": True,
        "name": GATEWAY_NAME,
        "public_base_url": public_base,
        "gateway_url": f"{public_base}/v1",
        "api_key": DEFAULT_API_KEY,
        "layers": _stack_layers(list(health_items)),
        "models": [_public_model(item) for item in models],
        "health": list(health_items),
        "integrations": _integration_snippets(public_base),
        "references": registry.get("stack", {}).get("reference_projects", []),
    }


@app.get("/v1/models")
async def gateway_models(request: Request) -> Dict[str, Any]:
    _check_gateway_key(request)
    return {
        "object": "list",
        "data": [
            {
                "id": item.get("id"),
                "object": "model",
                "created": 0,
                "owned_by": item.get("runner") or "local",
            }
            for item in _models()
        ],
    }


@app.post("/v1/chat/completions")
async def gateway_chat_completions(request: Request) -> Any:
    _check_gateway_key(request)
    payload = await request.json()
    model = _find_model("chat", payload.get("model"))
    payload["model"] = model["id"]
    if payload.get("stream"):
        return StreamingResponse(_stream_upstream(model, "chat/completions", payload), media_type="text/event-stream")
    data = await _post_json(model, "chat/completions", payload)
    return JSONResponse(content=data)


@app.post("/v1/images/generations")
async def gateway_image_generations(request: Request) -> Any:
    _check_gateway_key(request)
    payload = await request.json()
    model = _find_model("image", payload.get("model"))
    payload["model"] = model["id"]
    data = await _post_json(model, "images/generations", payload)
    return JSONResponse(content=data)


@app.post("/api/chat")
async def chat(request: ChatRequest) -> Dict[str, Any]:
    model = _find_model("chat", request.model)
    defaults = model.get("defaults") or {}
    temperature = request.temperature if request.temperature is not None else defaults.get("temperature", 0.7)
    max_tokens = request.max_tokens if request.max_tokens is not None else defaults.get("max_tokens", 1024)
    enable_thinking = request.enable_thinking
    if enable_thinking is None:
        enable_thinking = bool(defaults.get("enable_thinking", False))

    extra_body = dict(request.extra_body or {})
    chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
    chat_template_kwargs["enable_thinking"] = bool(enable_thinking)
    extra_body["chat_template_kwargs"] = chat_template_kwargs

    payload = {
        "model": model["id"],
        "messages": [{"role": item.role, "content": item.content} for item in request.messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_body": extra_body,
    }
    data = await _post_json(model, "chat/completions", payload)
    choices = data.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    return {
        "model": model["id"],
        "content": message.get("content", ""),
        "usage": data.get("usage"),
        "raw": data,
    }


@app.post("/api/images")
async def images(request: ImageRequest) -> Dict[str, Any]:
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    model = _find_model("image", request.model)
    defaults = model.get("defaults") or {}
    size = request.size or defaults.get("size", "512x512")
    steps = request.steps if request.steps is not None else defaults.get("steps", 12)
    true_cfg_scale = request.true_cfg_scale if request.true_cfg_scale is not None else defaults.get("true_cfg_scale", 4.0)

    extra_body = dict(request.extra_body or {})
    extra_body.update(
        {
            "num_inference_steps": int(steps),
            "true_cfg_scale": float(true_cfg_scale),
            "add_magic_prompt": bool(request.add_magic_prompt),
        }
    )

    payload: Dict[str, Any] = {
        "model": model["id"],
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt or " ",
        "size": size,
        "n": request.n,
        "response_format": "b64_json",
        "extra_body": extra_body,
    }
    if request.seed is not None:
        payload["seed"] = int(request.seed)

    data = await _post_json(model, "images/generations", payload)
    return {"model": model["id"], "data": data.get("data", []), "raw": data}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_ui:app", host=HOST, port=PORT, reload=False)
