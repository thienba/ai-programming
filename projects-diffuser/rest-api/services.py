import torch

from models import ImageRequest
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL.Image import Image

pipeline = StableDiffusionPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-v1-5",
    use_safetensors=True,
)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
    pipeline.scheduler.config)

if torch.cuda.is_available():
    pipeline.to("cuda")
elif torch.backends.mps.is_available():
    pipeline.to("mps")
else:
    pipeline.to("cpu")

async def generate_image(imgRequest: ImageRequest) -> Image:
    image: Image = pipeline(
        prompt=imgRequest.prompt,
        negative_prompt=imgRequest.negative_prompt,
        width=imgRequest.width,
        height=imgRequest.height,
        guidance_scale=imgRequest.guidance_scale,
        num_inference_steps=imgRequest.num_inference_steps,
    ).images[0]
    return image