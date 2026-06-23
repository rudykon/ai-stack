# ComfyUI / LocalAI Expansion Notes

The current image backend is `proxy_qwen_image.py` plus `qwen_image_worker.py`. It is intentionally small and lazy-loads Qwen-Image.

Use ComfyUI when you need:

- queued image jobs and visible progress
- reusable node workflows
- image editing workflows
- multiple image models and LoRAs

Use LocalAI when you need:

- one OpenAI-compatible service for several local modalities
- a broader local AI API surface beyond this project's Qwen-only setup
- model backends that can be swapped without changing client code

A future bridge can register either service in `model_registry.json` by adding another item with `type: "image"`, `base_url`, `api_key_env`, and `health_url`.
