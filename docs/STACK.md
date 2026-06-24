# Local AI Stack Architecture

This project keeps the lightweight local runner that already works on this machine, then adds the operational layers seen in mature community projects.

## Layers

1. UI layer
   - Current: `web/` + `web_ui.py`
   - Reference: Open WebUI
   - Added here: Stack tab, health cards, copyable integration snippets

2. Gateway layer
   - Current: `web_ui.py` exposes `/v1/models`, `/v1/chat/completions`, and `/v1/images/generations`
   - Reference: LiteLLM
   - Added here: model routing through `model_registry.json`, CORS, gateway API key check

3. LLM serving layer
   - Current: `proxy_qwen36.py` lazy-starts `serve_qwen36.py`
   - Reference target: vLLM or SGLang
   - Reason: keep the P40-compatible path while leaving a clear high-throughput replacement boundary

4. Image workflow layer
   - Current: `proxy_qwen_image.py` lazy-starts `qwen_image_worker.py`
   - Reference target: ComfyUI or LocalAI
   - Reason: current endpoint is simple and local; workflow engines can be added as extra registry entries later

5. Operations layer
   - Current: `logs/`, `outputs/`, `/api/stack`, `/health`
   - Added here: visible status across proxies and workers without forcing model load

## Gateway Contract

Use this single base URL for external clients:

```text
http://<LAN_IP>:8080/v1
```

The gateway accepts the same API key as the existing proxies:

```text
local-dev-key
```

Supported endpoints:

```text
GET  /v1/models
POST /v1/chat/completions
POST /v1/images/generations
```

## Extension Pattern

Add another backend by extending `model_registry.json`:

```json
{
  "id": "provider/model-name",
  "label": "Display name",
  "type": "chat",
  "runner": "vllm",
  "base_url": "http://127.0.0.1:8100/v1",
  "health_url": "http://127.0.0.1:8100/health",
  "api_key_env": "API_KEY",
  "default": false,
  "features": ["openai-chat", "streaming"]
}
```

For an image workflow backend, use `type: "image"` and the OpenAI Images API surface.
