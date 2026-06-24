#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export IMAGE_MODEL_PATH="${IMAGE_MODEL_PATH:-$ROOT/models/Qwen-Image-2512}"
export IMAGE_MODEL_NAME="${IMAGE_MODEL_NAME:-Qwen/Qwen-Image-2512}"
export API_KEY="${API_KEY:-local-dev-key}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8001}"
export IMAGE_WORKER_PORT="${IMAGE_WORKER_PORT:-18001}"
export IMAGE_IDLE_UNLOAD_SECONDS="${IMAGE_IDLE_UNLOAD_SECONDS:-300}"
export IMAGE_DEVICE_MAP="${IMAGE_DEVICE_MAP:-sequential}"
export IMAGE_DTYPE="${IMAGE_DTYPE:-float32}"
export IMAGE_GPU_MAX_MEMORY="${IMAGE_GPU_MAX_MEMORY:-20GiB}"
export IMAGE_CPU_MAX_MEMORY="${IMAGE_CPU_MAX_MEMORY:-120GiB}"
export IMAGE_DEFAULT_SIZE="${IMAGE_DEFAULT_SIZE:-1024x1024}"
export IMAGE_DEFAULT_STEPS="${IMAGE_DEFAULT_STEPS:-30}"

exec "$ROOT/.venv/bin/python" "$ROOT/proxy_qwen_image.py"
