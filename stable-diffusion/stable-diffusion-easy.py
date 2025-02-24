from diffusers import AutoencoderKL, StableDiffusionPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Adjust torch_dtype based on the device
torch_dtype = torch.float16 if device == "cuda" else torch.float32

# Load VAE và pipeline cho StableDiffusion
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", torch_dtype=torch_dtype).to(device)
# Tải model StableDiffusion về, hơi lâu chút nha
pipe = StableDiffusionPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-v1-5", vae=vae,
    torch_dtype=torch_dtype, variant="fp16"
).to(device)

# Bắt đầu tạo ảnh
puppy_image = pipe("A photograph of a puppy").images[0]

# Hiển thị ảnh trên jupiter notebook
# display(puppy_image)
# Hiển thị ảnh nếu chạy local
puppy_image.show()