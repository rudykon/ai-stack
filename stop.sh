#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ -f logs/proxy.pid ]; then
  pid="$(cat logs/proxy.pid)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
  fi
  rm -f logs/proxy.pid
fi

pkill -f "$ROOT/proxy_qwen36.py" 2>/dev/null || true
pkill -f "$ROOT/serve_qwen36.py" 2>/dev/null || true
