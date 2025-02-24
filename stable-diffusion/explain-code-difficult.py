from diffusers import AutoencoderKL, StableDiffusionPipeline
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load VAE và pipeline cho StableDiffusion
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-ema", torch_dtype=torch.float16
).to(device)
pipe = StableDiffusionPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-v1-5",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
).to(device)

# Implement hàm show_latent để chuyển đổi ảnh -> latents rồi hiển thị ảnh cho dễ hiểu


def show_latent(input_latents):
    latents = 1 / vae.config.scaling_factor * input_latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].show()


prompt = "A photograph of a puppy"
height = 512
width = 512
num_inference_steps = 30  # Denoising steps
guidance_scale = 7.5  # Scale for classifier-free guidance

# Tokenize the prompt đầu vào
text_input = pipe.tokenizer(
    prompt,
    padding="max_length",
    max_length=pipe.tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)

# Tạo thêm 1 chuỗi rỗng
negative_input = pipe.tokenizer(
    "",
    padding="max_length",
    max_length=pipe.tokenizer.model_max_length,
    return_tensors="pt",
)

# Tạo text embeddings từ tokenized text_input và negative_input
with torch.no_grad():
    text_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]
    negative_embeddings = pipe.text_encoder(
        negative_input.input_ids.to(device))[0]

# Gộp cả 2 lại thành text_embeddings để đưa vào cho unet
text_embeddings = torch.cat([negative_embeddings, text_embeddings])
print(text_embeddings)

# Tạo một latent ngẫu nhiên với `torch.randn`, có kích cỡ 4 * 64 * 64
latents = (
    torch.randn(
        (1, pipe.unet.config.in_channels, height // 8, width // 8),
    )
    .to(device)
    .half()
)

show_latent(latents)

# Set số bước sẽ chạy là num_inference_steps
pipe.scheduler.set_timesteps(num_inference_steps)

for i, t in enumerate(pipe.scheduler.timesteps):
    # Copy latents ra làm 2 phần
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = pipe.scheduler.scale_model_input(
        latent_model_input, t)

    # Dự đoán số lượng noise/nhiễu ở mỗi step, dựa theo latents và text_embeddings
    with torch.no_grad():
        noise_pred = pipe.unet(
            latents, t, encoder_hidden_states=text_embeddings
        ).sample

    # Impement classifier-free guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    # Trừ đi noise để khử nhiễu dần dần trong ảnh x_t -> x_t-1
    latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # Hiển thị ảnh sau mỗi bước khử nhiễu
    show_latent(latents)

# Dùng VAE để scale và decode the latents thành ảnh thức tế
# Hàm này giống hàm show_latents phía trên á
latents = 1 / vae.config.scaling_factor * latents
with torch.no_grad():
    image = vae.decode(latents).sample

image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]
pil_images[0].show()