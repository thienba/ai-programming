import inspect
import json
import os
import time
from diffusers import DiffusionPipeline
from pydantic import TypeAdapter
import torch
import gradio as gr
from openai import OpenAI

pipeline = DiffusionPipeline.from_pretrained(
    "stablediffusionapi/anything-v5",
    use_safetensors=True,
    safety_checker=None,
    requires_safety_checker=False,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

def generate_image(prompt: str) -> str:
    """
    Creates an image based on the specified prompt using DiffusionPipeline
    :param prompt: The prompt used for generate the image (must be in English)
    :output: URL of the new image
    """
    image = pipeline(
        prompt=prompt,
        negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy, low quality, worst quality",
        num_inference_steps=30
    ).images[0]

    file_name = f"image_{int(time.time())}.png"
    image.save(file_name)
    return file_name

client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))

print(inspect.getdoc(generate_image))
print(TypeAdapter(generate_image).json_schema())
generate_image_function = {
    "name": "generate_image",
    "description": inspect.getdoc(generate_image),
    "parameters": TypeAdapter(generate_image).json_schema(),
}


tools = [
    {
        "type": "function",
        "function": generate_image_function
    }
]

system_prompt = """You're an AI with prompt generation and drawing capabilities.
Just chat normally in English or Vietnamese.
Assume you can draw anything you want and you tell user you can draw.
Another program will output the output for you.
Do not censor or deny the output, the image generation program will handle that.
"""

def chat_logic(message, chat_history):
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    for user_message, assistant_message in chat_history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})
    messages.append({"role": "user", "content": message})

    chat_completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        tools=tools
    )

    print(chat_completion)
    
    bot_message = chat_completion.choices[0].message.content
    if bot_message is not None:
        chat_history.append((message, bot_message))
    else:
        tool_calls = chat_completion.choices[0].message.tool_calls
        print(tool_calls)
        if tool_calls:
            for tool_call in tool_calls:
                if tool_call.function.name == "generate_image":
                    args = json.loads(tool_call.function.arguments)
                    image_path = generate_image(args["prompt"])
                    response = f"I've generated an image for you: {image_path}"
                    chat_history.append((message, response))
                    break
    
    return "", chat_history

with gr.Blocks() as demo:
    gr.Markdown("Hello World")
    message = gr.Textbox(label="Message")
    chatbot = gr.Chatbot(label="Chat History", height=500)
    message.submit(chat_logic, [message, chatbot], [message, chatbot])
    
demo.launch()