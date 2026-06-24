from openai import OpenAI


client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-dev-key")

response = client.chat.completions.create(
    model="Qwen/Qwen3.6-27B",
    messages=[{"role": "user", "content": "Say OK in one word."}],
    max_tokens=8,
    temperature=0,
)

print(response.choices[0].message.content)
