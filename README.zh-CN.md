# 本地 AI Stack

[English](README.md) | 中文

![本地 AI Stack 架构](docs/assets/stack-architecture.svg)

这是一个面向局域网和单机 GPU 的本地 AI 工作台。当前默认集成 Qwen 文本模型和 Qwen-Image 文生图模型，同时保留通用 OpenAI-compatible 网关、模型注册中心、轻量 Web UI 和显存友好的懒加载机制，方便后续接入其他大模型。

它的目标不是替代 Open WebUI、ComfyUI、vLLM、SGLang 或 LiteLLM，而是在当前机器上提供一个更轻、更直接可用、便于二次开发的本地服务底座。

## 主要特色

- **局域网优先**：默认通过 `http://<LAN_IP>:8080` 访问 Web 控制台。
- **统一 OpenAI-compatible 网关**：外部工具可以只接 `http://<LAN_IP>:8080/v1`。
- **文本与文生图并存**：同时提供 Qwen3.6-27B 聊天和 Qwen-Image-2512 图片生成。
- **显存友好的懒加载**：模型 worker 首次请求才启动，空闲后自动退出释放 CUDA 显存。
- **Stack 可视化**：Web UI 里有 Stack 页面，可查看 UI、Gateway、LLM Serving、Image Workflow 和 Operations 状态。
- **可扩展服务注册中心**：通过 `model_registry.json` 注册更多 OpenAI-compatible 后端。
- **面向本机硬件的取舍**：图片模型默认使用适合 Tesla P40 的 `float32 + sequential offload` 配置。

## 界面示意

![控制台示意](docs/assets/console-overview.svg)

## 快速启动

```bash
./api.sh
```

启动后访问：

```text
Web UI：http://127.0.0.1:8080
局域网 Web UI：http://<LAN_IP>:8080
统一网关：http://<LAN_IP>:8080/v1
API key：local-dev-key
```

把示例里的 `<LAN_IP>` 替换为 `./api.sh` 启动时打印的局域网地址。保持这个终端打开。按 `Ctrl+C` 会停止 Web UI、文本 API、图片 API，以及它们启动的 worker。

常用命令：

```bash
./api.sh status
./api.sh stop
```

## 请求流转

![请求流转](docs/assets/request-flow.svg)

统一网关支持：

```text
GET  /v1/models
POST /v1/chat/completions
POST /v1/images/generations
```

示例：

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer local-dev-key' \
  -d '{
    "model": "Qwen/Qwen3.6-27B",
    "messages": [{"role": "user", "content": "Say OK in one word."}],
    "max_tokens": 8,
    "temperature": 0
  }'
```

图片生成示例：

```bash
curl http://127.0.0.1:8080/v1/images/generations \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer local-dev-key' \
  -d '{
    "model": "Qwen/Qwen-Image-2512",
    "prompt": "A red circle on a white background",
    "size": "512x512",
    "n": 1,
    "response_format": "b64_json"
  }'
```

## 项目结构

```text
api.sh                    一键启动 Web UI、文本代理和图片代理
web_ui.py                 Web UI 后端，同时提供统一 /v1 网关
web/                      浏览器控制台前端
model_registry.json       模型、runner、health、features 注册中心
proxy_qwen36.py           文本 API 懒加载代理，默认端口 8000
serve_qwen36.py           Qwen3.6-27B 文本 worker
proxy_qwen_image.py       图片 API 懒加载代理，默认端口 8001
qwen_image_worker.py      Qwen-Image worker
models/                   本地模型权重，默认不上传 GitHub
outputs/                  生成图片输出，默认不上传 GitHub
logs/                     运行日志，默认不上传 GitHub
docs/STACK.md             分层架构说明
integrations/             Open WebUI、LiteLLM、ComfyUI/LocalAI 接入说明
references/               社区项目 README 快照
```

## 文本模型 API

文本模型通过两层服务提供：

- 外层代理：`proxy_qwen36.py`，端口 `8000`
- 内部 worker：`serve_qwen36.py`，端口 `18000`

直连文本 API：

```text
http://127.0.0.1:8000/v1
http://<LAN_IP>:8000/v1
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

如果返回 `worker_running:false`，说明模型 worker 没有驻留，显存应该接近空闲状态。

## 图片模型 API

图片模型同样通过懒加载代理提供：

- 外层代理：`proxy_qwen_image.py`，端口 `8001`
- 内部 worker：`qwen_image_worker.py`，端口 `18001`

直连图片 API：

```text
http://127.0.0.1:8001/v1
http://<LAN_IP>:8001/v1
```

健康检查：

```bash
curl http://127.0.0.1:8001/health
```

生成的图片会保存到：

```text
outputs/qwen-image/
```

## 模型注册

模型选择来自 `model_registry.json`。新增一个 OpenAI-compatible 聊天模型时，可以加入：

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

图片后端使用 `"type": "image"`，并提供 OpenAI Images API 兼容接口即可。

## 与社区项目的关系

本项目借鉴的是分层思路，而不是复制大型项目代码：

- Open WebUI：成熟 Web UI、多模型入口和 OpenAI-compatible 客户端体验。
- LiteLLM：统一网关、模型路由和外部工具接入方式。
- vLLM / SGLang：未来可替换的高吞吐文本推理层。
- ComfyUI / LocalAI：未来可扩展的图片工作流和多模态服务层。

相关说明在：

```text
docs/STACK.md
integrations/
references/
```

## 验证

不触发大模型加载的轻量验证：

```bash
curl http://127.0.0.1:8080/api/health
curl http://127.0.0.1:8080/api/stack
curl http://127.0.0.1:8080/v1/models -H 'Authorization: Bearer local-dev-key'
```

会触发文本模型加载的冒烟测试：

```bash
.venv/bin/python smoke_test_gateway.py
```

会触发图片模型加载的冒烟测试：

```bash
.venv/bin/python smoke_test_qwen_image.py
```

## 注意事项

- `models/`、`.venv/`、`logs/`、`outputs/` 和 `github_token.json` 已在 `.gitignore` 中排除。
- 当前默认 API key 是开发用固定值，局域网之外使用前应改成强密钥。
- Qwen-Image 在 Tesla P40 上默认使用 `IMAGE_DTYPE=float32` 和 `IMAGE_DEVICE_MAP=sequential`，这是为了避免 FP16 黑图问题。
- 空闲释放显存依赖终止 worker 进程，这是当前硬件上最可靠的方式。
