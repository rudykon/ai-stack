# Integration Notes

This project now exposes two integration surfaces:

- Web console: `http://<lan-ip>:8080`
- Unified OpenAI-compatible gateway: `http://<lan-ip>:8080/v1`

The gateway routes chat requests to the local text proxy on `8000` and image requests to the local image proxy on `8001`. Use `API_KEY` from the runtime environment, defaulting to `local-dev-key`.

Files in this directory are examples. They do not start external projects by themselves.
