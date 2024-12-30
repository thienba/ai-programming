import gradio as gr
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load the processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Define the function to generate captions
def generate_caption(raw_image: Image) -> str:
    inputs = processor(raw_image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Create a Gradio interface
iface = gr.Interface(
    fn=generate_caption,  # Function to call
    inputs=['image'],  # Input component
    outputs=['text'],  # Output component
    title="Image Captioning",  # Title of the interface
    description="Upload an image to generate a caption."  # Description
)

iface.launch()