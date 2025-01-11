# -*- coding: utf-8 -*-
"""Learn OpenAI SDK - GroqAI

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1bqGLZiYW8LHWxddpd9nR_D_cvYO-T9Id

## **Nhớ bấm ô trên cùng để cài openai nha!**
"""

# !pip install openai

from openai import OpenAI

# Nếu các bạn dùng GroqAI
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    # Làm theo hướng dẫn trong bài, truy cập https://console.groq.com/keys để lấy API Key nha
    api_key='gsk_XXXXX',
)

chat_completion = client.chat.completions.create(
   messages=[
        {
	    # Cài đặt cách trả lời, nhiệm vụ của bot
            "role": "system",
            "content": "You are a friendly and flirty female tour guide.",
        },
        {
	    # Câu hỏi gốc của bạn
            "role": "user",
            "content": "Thủ đô của Pháp là gì?",
        },
        {
	    # Câu trả lời của bot
            "role": "assistant",
            "content": "Thủ đô của Pháp là Paris.",
        },
        {
	    # Câu hỏi tiếp theo của bạn
            "role": "user",
            "content": "Tới đó rồi thì nên đi đâu chơi?",
        }
    ],
    model="gemma2-9b-it",
)

print(chat_completion.choices[0].message.content)

stream = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Hello. Who are you? Write me a long poem to introduce your self.",
        }
    ],
    model="gemma2-9b-it",
    stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")