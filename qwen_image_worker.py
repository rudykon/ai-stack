import base64
import gc
import os
import time
import uuid
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("IMAGE_MODEL_PATH", ROOT / "models" / "Qwen-Image-2512"))
MODEL_NAME = os.getenv("IMAGE_MODEL_NAME", "Qwen/Qwen-Image-2512")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "18001"))
IMAGE_DEVICE_MAP = os.getenv("IMAGE_DEVICE_MAP", "balanced")
IMAGE_DTYPE = os.getenv("IMAGE_DTYPE", "auto").strip().lower()
IMAGE_GPU_MAX_MEMORY = os.getenv("IMAGE_GPU_MAX_MEMORY", "20GiB")
IMAGE_CPU_MAX_MEMORY = os.getenv("IMAGE_CPU_MAX_MEMORY", "120GiB")
DEFAULT_SIZE = os.getenv("IMAGE_DEFAULT_SIZE", "1024x1024")
DEFAULT_STEPS = int(os.getenv("IMAGE_DEFAULT_STEPS", "30"))
DEFAULT_TRUE_CFG_SCALE = float(os.getenv("IMAGE_DEFAULT_TRUE_CFG_SCALE", "4.0"))
MAX_IMAGES = int(os.getenv("IMAGE_MAX_BATCH", "2"))
OUTPUT_DIR = Path(os.getenv("IMAGE_OUTPUT_DIR", ROOT / "outputs" / "qwen-image"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Qwen-Image worker", version="0.1.0")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

_pipe = None
_load_error: Optional[str] = None
_generation_lock = Lock()


class ImageGenerationRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    negative_prompt: Optional[str] = " "
    n: int = 1
    size: Optional[str] = DEFAULT_SIZE
    width: Optional[int] = None
    height: Optional[int] = None
    response_format: Optional[str] = "b64_json"
    seed: Optional[int] = None
    num_inference_steps: Optional[int] = None
    true_cfg_scale: Optional[float] = None
    max_sequence_length: Optional[int] = 512
    extra_body: Dict[str, Any] = Field(default_factory=dict)


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _parse_size(request: ImageGenerationRequest) -> Tuple[int, int]:
    if request.width and request.height:
        width, height = request.width, request.height
    else:
        size = request.size or DEFAULT_SIZE
        if "x" not in size:
            raise HTTPException(status_code=400, detail="size must look like 1024x1024")
        left, right = size.lower().split("x", 1)
        width, height = int(left), int(right)
    if width < 256 or height < 256 or width > 2048 or height > 2048:
        raise HTTPException(status_code=400, detail="width and height must be between 256 and 2048")
    if width % 16 or height % 16:
        raise HTTPException(status_code=400, detail="width and height must be multiples of 16")
    return width, height


def _torch_dtype() -> torch.dtype:
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if IMAGE_DTYPE in aliases:
        return aliases[IMAGE_DTYPE]
    if IMAGE_DTYPE != "auto":
        raise HTTPException(status_code=500, detail=f"Unsupported IMAGE_DTYPE={IMAGE_DTYPE}")
    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability(0)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def _max_memory() -> Dict[Any, str]:
    if not torch.cuda.is_available():
        return {"cpu": IMAGE_CPU_MAX_MEMORY}
    return {i: IMAGE_GPU_MAX_MEMORY for i in range(torch.cuda.device_count())} | {"cpu": IMAGE_CPU_MAX_MEMORY}


def _load_pipeline() -> None:
    global _pipe, _load_error
    if _pipe is not None:
        return
    if not (MODEL_PATH / "model_index.json").exists():
        raise HTTPException(status_code=503, detail=f"Qwen-Image is not downloaded at {MODEL_PATH}. Run download_qwen_image.py first.")

    from diffusers import QwenImagePipeline

    dtype = _torch_dtype()
    kwargs = {
        "torch_dtype": dtype,
        "local_files_only": True,
        "use_safetensors": True,
    }
    if torch.cuda.is_available() and IMAGE_DEVICE_MAP.lower() not in {"", "none", "single", "sequential", "offload", "cpu_offload"}:
        kwargs["device_map"] = IMAGE_DEVICE_MAP
        kwargs["max_memory"] = _max_memory()

    try:
        _pipe = QwenImagePipeline.from_pretrained(str(MODEL_PATH), **kwargs)
    except Exception as exc:
        _load_error = f"device_map load failed: {exc}"
        fallback_kwargs = {
            "torch_dtype": dtype,
            "local_files_only": True,
            "use_safetensors": True,
        }
        _pipe = QwenImagePipeline.from_pretrained(str(MODEL_PATH), **fallback_kwargs)
        if torch.cuda.is_available():
            _pipe.enable_sequential_cpu_offload(gpu_id=0)
        else:
            _pipe = _pipe.to("cpu")
    else:
        if torch.cuda.is_available() and IMAGE_DEVICE_MAP.lower() in {"sequential", "offload", "cpu_offload"}:
            _pipe.enable_sequential_cpu_offload(gpu_id=0)
        elif torch.cuda.is_available() and "device_map" not in kwargs:
            _pipe = _pipe.to("cuda")

    if dtype == torch.float16 and hasattr(_pipe, "vae"):
        _pipe.vae.to(dtype=torch.float32)

    if hasattr(_pipe, "enable_vae_tiling"):
        _pipe.enable_vae_tiling()
    if hasattr(_pipe, "set_progress_bar_config"):
        _pipe.set_progress_bar_config(disable=True)


def _encode_image(image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _save_image(image, seed: Optional[int]) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = f"seed{seed}" if seed is not None else uuid.uuid4().hex[:8]
    path = OUTPUT_DIR / f"qwen-image-{stamp}-{suffix}.png"
    image.save(path)
    return path.name


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "model": MODEL_NAME,
        "loaded": _pipe is not None,
        "model_path": str(MODEL_PATH),
        "device_map": IMAGE_DEVICE_MAP,
        "dtype": str(_torch_dtype()).replace("torch.", ""),
        "last_load_error": _load_error,
        "cuda": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


@app.post("/v1/images/generations")
def generate(request: ImageGenerationRequest) -> Dict[str, Any]:
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")
    if request.n < 1 or request.n > MAX_IMAGES:
        raise HTTPException(status_code=400, detail=f"n must be between 1 and {MAX_IMAGES}")

    with _generation_lock:
        _load_pipeline()
        width, height = _parse_size(request)
        extra = request.extra_body or {}
        prompt = request.prompt
        add_magic_prompt = bool(extra.get("add_magic_prompt", False))
        if add_magic_prompt:
            prompt += ", 超清，4K，电影级构图." if _has_cjk(prompt) else ", Ultra HD, 4K, cinematic composition."

        steps = int(request.num_inference_steps or extra.get("num_inference_steps") or DEFAULT_STEPS)
        true_cfg_scale = float(request.true_cfg_scale or extra.get("true_cfg_scale") or DEFAULT_TRUE_CFG_SCALE)
        seed = request.seed if request.seed is not None else extra.get("seed")
        max_sequence_length = int(request.max_sequence_length or extra.get("max_sequence_length") or 512)

        data: List[Dict[str, Any]] = []
        for index in range(request.n):
            current_seed = None if seed is None else int(seed) + index
            generator = None
            if current_seed is not None:
                generator = torch.Generator(device="cpu").manual_seed(current_seed)
            with torch.inference_mode():
                result = _pipe(
                    prompt=prompt,
                    negative_prompt=request.negative_prompt or " ",
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    true_cfg_scale=true_cfg_scale,
                    generator=generator,
                    max_sequence_length=max_sequence_length,
                )
            image = result.images[0]
            filename = _save_image(image, current_seed)
            item: Dict[str, Any] = {
                "url": f"/outputs/{filename}",
                "seed": current_seed,
                "width": width,
                "height": height,
            }
            if request.response_format == "b64_json":
                item["b64_json"] = _encode_image(image)
            data.append(item)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {"created": int(time.time()), "data": data}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("qwen_image_worker:app", host=HOST, port=PORT, reload=False)
