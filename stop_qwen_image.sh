#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

systemctl --user stop qwen-image-api.service 2>/dev/null || true
pkill -f "$ROOT/proxy_qwen_image.py" 2>/dev/null || true
pkill -f "$ROOT/qwen_image_worker.py" 2>/dev/null || true
