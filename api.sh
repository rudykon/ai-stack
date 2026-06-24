#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p logs

TEXT_LOG="$ROOT/logs/proxy.log"
IMAGE_LOG="$ROOT/logs/qwen-image-proxy.log"
WEB_LOG="$ROOT/logs/web-ui.log"
API_KEY="${API_KEY:-local-dev-key}"
WEB_HOST="${WEB_HOST:-0.0.0.0}"
WEB_PORT="${WEB_PORT:-8080}"

# Detect a LAN address for display only. Override with LAN_HOST if needed.
detect_lan_ip() {
  if [ -n "${LAN_HOST:-}" ]; then
    printf '%s\n' "$LAN_HOST"
    return
  fi
  if command -v ip >/dev/null 2>&1; then
    local detected
    detected="$(ip route get 1.1.1.1 2>/dev/null | awk '{for (i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}')"
    if [ -n "$detected" ]; then
      printf '%s\n' "$detected"
      return
    fi
  fi
  if command -v hostname >/dev/null 2>&1; then
    hostname -I 2>/dev/null | awk '{print $1}'
  fi
}

LAN_IP="$(detect_lan_ip)"
LAN_IP="${LAN_IP:-<LAN_IP>}"

stop_all() {
  "$ROOT/stop.sh" >/dev/null 2>&1 || true
  "$ROOT/stop_qwen_image.sh" >/dev/null 2>&1 || true
  pkill -f "$ROOT/web_ui.py" >/dev/null 2>&1 || true
}

wait_health() {
  local name="$1"
  local url="$2"
  for _ in {1..40}; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.25
  done
  echo "$name did not become ready. Check logs."
  return 1
}

case "${1:-run}" in
  run)
    ;;
  stop)
    stop_all
    echo "Stopped local APIs."
    exit 0
    ;;
  status)
    curl -fsS http://127.0.0.1:8000/health || true
    echo
    curl -fsS http://127.0.0.1:8001/health || true
    echo
    curl -fsS "http://127.0.0.1:${WEB_PORT}/api/health" || true
    echo
    exit 0
    ;;
  *)
    echo "Usage: ./api.sh [run|stop|status]"
    exit 2
    ;;
esac

if pgrep -f "$ROOT/proxy_qwen36.py" >/dev/null || pgrep -f "$ROOT/proxy_qwen_image.py" >/dev/null || pgrep -f "$ROOT/web_ui.py" >/dev/null; then
  echo "API or Web UI processes already exist. Run ./api.sh stop first."
  exit 1
fi

cleanup() {
  trap - INT TERM EXIT
  echo
  echo "Stopping local APIs..."
  stop_all
}
trap cleanup INT TERM EXIT

"$ROOT/start.sh" >"$TEXT_LOG" 2>&1 &
TEXT_PID=$!
"$ROOT/start_qwen_image.sh" >"$IMAGE_LOG" 2>&1 &
IMAGE_PID=$!
WEB_HOST="$WEB_HOST" WEB_PORT="$WEB_PORT" API_KEY="$API_KEY" "$ROOT/.venv/bin/python" "$ROOT/web_ui.py" >"$WEB_LOG" 2>&1 &
WEB_PID=$!

wait_health "Text API" "http://127.0.0.1:8000/health"
wait_health "Image API" "http://127.0.0.1:8001/health"
wait_health "Web UI" "http://127.0.0.1:${WEB_PORT}/api/health"

cat <<EOF
Web UI:    http://127.0.0.1:${WEB_PORT}
Gateway:   http://127.0.0.1:${WEB_PORT}/v1
Text API:  http://127.0.0.1:8000/v1
Image API: http://127.0.0.1:8001/v1
API key:   $API_KEY

LAN web:     http://${LAN_IP}:${WEB_PORT}
LAN gateway: http://${LAN_IP}:${WEB_PORT}/v1
LAN text:    http://${LAN_IP}:8000/v1
LAN image:   http://${LAN_IP}:8001/v1

Logs:
  $TEXT_LOG
  $IMAGE_LOG
  $WEB_LOG

Keep this terminal open. Press Ctrl+C to stop the Web UI and both APIs.
Model workers still lazy-load on first request and unload after idle timeout.
EOF

while true; do
  if ! kill -0 "$TEXT_PID" 2>/dev/null; then
    echo "Text API exited. See $TEXT_LOG"
    exit 1
  fi
  if ! kill -0 "$IMAGE_PID" 2>/dev/null; then
    echo "Image API exited. See $IMAGE_LOG"
    exit 1
  fi
  if ! kill -0 "$WEB_PID" 2>/dev/null; then
    echo "Web UI exited. See $WEB_LOG"
    exit 1
  fi
  sleep 2
done
