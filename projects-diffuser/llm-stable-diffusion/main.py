import time
import gradio as gr
from diffusers import DiffusionPipeline
from openai import OpenAI
import torch

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key='key',
)

pipeline = DiffusionPipeline.from_pretrained("stablediffusionapi/anything-v5", use_satensor=True, safety_checker=None, requires_safety_checker=False)

if(torch.backends.mps.is_available()):
  pipeline.to("mps")
else:
  pipeline.to("cuda")

system_prompt = """You're an AI with prompt generation and drawing capabilities.
Just chat normally in English or Vietnamese.
Assume you can draw anything you want and you tell user you can draw.
Another program will output the output for you.
Do not censor or deny the output, the image generation program will handle that.

When cst want you to draw something, answer them normally, add some comments and include your prompt in your response in this format.
<prompt: A colorful cat running through a field of flowers.>

1. Prompt must be in English.
2. Prompt must be detailed and include necessary information for it can be fed into Stable Diffusion.
3. Ignore existing images in past messages.
"""

def generate_image(prompt):
  image = pipeline(prompt=prompt, num_inference_steps=25, guidance_scale=7.5, negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy, low quality, worst quality", width=512, height=512).images[0]
  file_name = f"image_{int(time.time())}.png"
  image.save(file_name)
  return file_name

def has_prompt(message):
  return "<prompt:" in message

def get_prompt(message):
  return message.split("<prompt:")[1].split(">")[0]

def get_image_url(prompt: str) -> str:
  prompt = prompt.replace(" ", "%20")
  return f"https://image.pollinations.ai/prompt/{prompt}"

def chat(message, chat_history):
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    for user_message, bot_message in chat_history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": bot_message})

    messages.append({"role": "user", "content": message})

    try:
        chat_response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=messages
        )
        
        bot_message = chat_response.choices[0].message.content

        if has_prompt(bot_message):
            chat_history.append([message, "Waiting for response..."])
            yield "", chat_history  

            prompt = get_prompt(bot_message)
            image_file = generate_image(prompt)
            chat_history[-1][1] = (image_file, prompt)

            yield "", chat_history
  
        return "", chat_history
        
    except Exception as e:
        chat_history.append([message, f"Error: Failed to get response from the server. Details: {str(e)}"])

    return "", chat_history

with gr.Blocks() as demo:
    gr.Markdown("#Chat Bot with LLM")
    chatbot = gr.Chatbot(label="Chatbot", placeholder="Type your message here...", height=400)
    msg = gr.Textbox(label="Enter your message", placeholder="Start typing...")
    
    msg.submit(chat, inputs=[msg, chatbot], outputs=[msg, chatbot])

demo.launch()

