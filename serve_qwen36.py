import asyncio
import gc
import json
import os
import time
import uuid
from threading import Thread
from typing import Any, Dict, Iterable, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoProcessor, AutoTokenizer, TextIteratorStreamer


ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(ROOT, "models", "Qwen3.6-27B"))
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3.6-27B")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
IDLE_UNLOAD_SECONDS = int(os.getenv("IDLE_UNLOAD_SECONDS", "300"))
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "32768"))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", "1024"))
DEVICE_MAP = os.getenv("DEVICE_MAP", "auto")
ATTN_IMPLEMENTATION = os.getenv("ATTN_IMPLEMENTATION", "eager")
GPU_MAX_MEMORY = os.getenv("GPU_MAX_MEMORY", "20GiB")
CPU_MAX_MEMORY = os.getenv("CPU_MAX_MEMORY", "120GiB")


app = FastAPI(title="Qwen3.6-27B local API", version="0.1.0")

_model = None
_tokenizer = None
_processor = None
_last_used = 0.0
_active_requests = 0
_load_lock = asyncio.Lock()
_generation_lock = asyncio.Lock()


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


def _gpu_max_memory() -> Dict[Any, str]:
    if not torch.cuda.is_available():
        return {"cpu": CPU_MAX_MEMORY}
    return {i: GPU_MAX_MEMORY for i in range(torch.cuda.device_count())} | {"cpu": CPU_MAX_MEMORY}


def _touch() -> None:
    global _last_used
    _last_used = time.monotonic()


def _contains_media(messages: Iterable[ChatMessage]) -> bool:
    for message in messages:
        if isinstance(message.content, list):
            for part in message.content:
                if isinstance(part, dict) and part.get("type") in {"image_url", "video_url", "input_image"}:
                    return True
    return False


def _normalize_messages(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    normalized = []
    for message in messages:
        content = message.content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
            content = "\n".join(piece for piece in text_parts if piece)
        normalized.append({"role": message.role, "content": content})
    return normalized


async def _load_model() -> None:
    global _model, _tokenizer, _processor
    if _model is not None:
        return

    async with _load_lock:
        if _model is not None:
            return

        if not os.path.exists(os.path.join(MODEL_PATH, "config.json")):
            raise HTTPException(
                status_code=503,
                detail=f"Model is not downloaded at {MODEL_PATH}. Run download_model.py first.",
            )

        from transformers import AutoModelForCausalLM

        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            AutoModelForImageTextToText = None

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        common_kwargs = {
            "torch_dtype": dtype,
            "device_map": DEVICE_MAP,
            "max_memory": _gpu_max_memory(),
            "trust_remote_code": True,
            "attn_implementation": ATTN_IMPLEMENTATION,
        }

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        try:
            _processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        except Exception:
            _processor = None

        candidates = []
        if AutoModelForImageTextToText is not None:
            candidates.append(AutoModelForImageTextToText)
        candidates.append(AutoModelForCausalLM)

        last_error = None
        for model_cls in candidates:
            for kwargs in (
                common_kwargs,
                {key: value for key, value in common_kwargs.items() if key != "attn_implementation"},
            ):
                try:
                    _model = model_cls.from_pretrained(MODEL_PATH, **kwargs)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
            if _model is not None:
                break
        if last_error is not None:
            raise last_error

        _model.eval()
        _touch()


def _unload_model() -> None:
    global _model, _tokenizer, _processor
    if _model is None:
        return
    _model = None
    _tokenizer = None
    _processor = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


async def _idle_unloader() -> None:
    while True:
        await asyncio.sleep(30)
        if _model is None or _active_requests:
            continue
        if time.monotonic() - _last_used >= IDLE_UNLOAD_SECONDS:
            _unload_model()


@app.on_event("startup")
async def startup_event() -> None:
    asyncio.create_task(_idle_unloader())


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "model": MODEL_NAME,
        "loaded": _model is not None,
        "idle_unload_seconds": IDLE_UNLOAD_SECONDS,
        "cuda": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


@app.get("/v1/models")
async def list_models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "created": 0, "owned_by": "local"}],
    }


def _generation_kwargs(request: ChatCompletionRequest, input_len: int) -> Dict[str, Any]:
    max_new_tokens = request.max_tokens or DEFAULT_MAX_NEW_TOKENS
    temperature = request.temperature if request.temperature is not None else 1.0
    top_p = request.top_p if request.top_p is not None else 0.95
    extra = request.extra_body or {}

    kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": _tokenizer.eos_token_id,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    if "top_k" in extra:
        kwargs["top_k"] = int(extra["top_k"])
    if request.repetition_penalty is not None:
        kwargs["repetition_penalty"] = request.repetition_penalty
    if input_len > MAX_INPUT_TOKENS:
        raise HTTPException(
            status_code=400,
            detail=f"Input has {input_len} tokens, over MAX_INPUT_TOKENS={MAX_INPUT_TOKENS}.",
        )
    return kwargs


def _encode_prompt(request: ChatCompletionRequest) -> Any:
    if _contains_media(request.messages):
        raise HTTPException(
            status_code=400,
            detail="This local lightweight wrapper is configured for text chat. Use transformers serve or vLLM for image/video inputs.",
        )

    extra = request.extra_body or {}
    chat_template_kwargs = extra.get("chat_template_kwargs") or {}
    messages = _normalize_messages(request.messages)
    prompt = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )
    inputs = _tokenizer([prompt], return_tensors="pt")
    if hasattr(_model, "device"):
        inputs = inputs.to(_model.device)
    return inputs


def _chat_completion_body(request: ChatCompletionRequest, content: str, prompt_tokens: int, completion_tokens: int) -> Dict[str, Any]:
    created = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created,
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> Any:
    global _active_requests
    await _load_model()
    _active_requests += 1
    _touch()

    if request.stream:
        return StreamingResponse(_stream_chat(request), media_type="text/event-stream")

    try:
        async with _generation_lock:
            inputs = _encode_prompt(request)
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            kwargs = _generation_kwargs(request, prompt_tokens)
            with torch.inference_mode():
                output = _model.generate(**inputs, **kwargs)
            new_tokens = output[0, prompt_tokens:]
            content = _tokenizer.decode(new_tokens, skip_special_tokens=True)
            completion_tokens = int(new_tokens.shape[-1])
            return _chat_completion_body(request, content, prompt_tokens, completion_tokens)
    finally:
        _active_requests -= 1
        _touch()


async def _stream_chat(request: ChatCompletionRequest) -> Any:
    global _active_requests
    try:
        async with _generation_lock:
            inputs = _encode_prompt(request)
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            kwargs = _generation_kwargs(request, prompt_tokens)
            streamer = TextIteratorStreamer(_tokenizer, skip_prompt=True, skip_special_tokens=True)
            kwargs["streamer"] = streamer
            created = int(time.time())
            chunk_id = f"chatcmpl-{uuid.uuid4().hex}"

            def run_generation() -> None:
                with torch.inference_mode():
                    _model.generate(**inputs, **kwargs)

            thread = Thread(target=run_generation, daemon=True)
            thread.start()
            first = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_NAME,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
            for text in streamer:
                chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_NAME,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            thread.join()
            final = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_NAME,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
    finally:
        _active_requests -= 1
        _touch()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve_qwen36:app", host=HOST, port=PORT, reload=False)
