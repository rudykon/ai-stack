#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p logs

if [ -f logs/proxy.pid ]; then
  old_pid="$(cat logs/proxy.pid)"
  if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
    echo "API service is already running with PID $old_pid"
    exit 0
  fi
fi

nohup ./start.sh > logs/proxy.log 2>&1 &
echo $! > logs/proxy.pid
echo "Started API service with PID $(cat logs/proxy.pid)"
