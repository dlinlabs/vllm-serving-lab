from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-4B-Instruct-2507",
    messages=[
        {"role": "user", "content": "Explain in one sentence what vLLM does."}
    ],
    max_tokens=100,
)

print(response.choices[0].message.content)
