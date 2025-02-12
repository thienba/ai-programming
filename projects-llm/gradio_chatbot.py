import gradio as gr
from openai import OpenAI

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key='api_key',
)

def chat(message, chat_history):
    # Convert chat_history to messages
    messages = []
    for user_message, bot_message in chat_history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": bot_message})

    # Add new user message to messages
    messages.append({"role": "user", "content": message})

    chat_history.append([message, "Waiting for response..."])
    yield "", chat_history  # Trả về ngay để hiển thị "Waiting..."

    # Sau đó tiếp tục chạy để lấy response
    try:
        # Get bot response
        chat_response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=messages,
            stream=True,
        )
    except Exception as e:
        chat_history[-1][1] = "Error: Failed to get response from the server."
        return "", chat_history

    # Replace the bot last message with an empty string
    chat_history[-1][1] = ""

    for chunk in chat_response:
        if chunk.choices[0].delta.content:
            chat_history[-1][1] += chunk.choices[0].delta.content
            yield "", chat_history

    return "", chat_history

with gr.Blocks() as demo:
    gr.Markdown("#Chat Bot with LLM")
    chatbot = gr.Chatbot(label="Chatbot", placeholder="Type your message here...", height=400)
    msg = gr.Textbox(label="Enter your message", placeholder="Start typing...")
    
    # Set up event handlers
    msg.submit(chat, inputs=[msg, chatbot], outputs=[msg, chatbot])

demo.launch()

