from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key='random_api_key',
)

stream = client.chat.completions.create(
    messages=[
        {
            "role": "user", 
            "content": "Hello. Who are you? Write me a long poem to introduce your self.",
        }
    ],
    model="deepseek-r1-distill-llama-8b",
    stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")