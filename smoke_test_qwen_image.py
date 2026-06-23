import base64
from pathlib import Path

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="EMPTY")

response = client.images.generate(
    model="Qwen/Qwen-Image-2512",
    prompt="A simple red circle on a white background, clean icon style.",
    size="256x256",
    n=1,
    response_format="b64_json",
    extra_body={"num_inference_steps": 4, "true_cfg_scale": 4.0, "seed": 42},
)

image_bytes = base64.b64decode(response.data[0].b64_json)
out = Path("outputs/qwen-image-smoke.png")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_bytes(image_bytes)
print(out)
