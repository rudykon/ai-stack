#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export MODEL_PATH="${MODEL_PATH:-$ROOT/models/Qwen3.6-27B}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B}"
export API_KEY="${API_KEY:-local-dev-key}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export IDLE_UNLOAD_SECONDS="${IDLE_UNLOAD_SECONDS:-300}"
export MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-32768}"
export DEFAULT_MAX_NEW_TOKENS="${DEFAULT_MAX_NEW_TOKENS:-1024}"
export GPU_MAX_MEMORY="${GPU_MAX_MEMORY:-20GiB}"
export CPU_MAX_MEMORY="${CPU_MAX_MEMORY:-120GiB}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"

exec "$ROOT/.venv/bin/python" "$ROOT/proxy_qwen36.py"
