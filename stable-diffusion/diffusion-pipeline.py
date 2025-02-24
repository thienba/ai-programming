from diffusers import DiffusionPipeline, EulerDiscreteScheduler, DDIMScheduler, DPMSolverMultistepScheduler
import torch

# Các bạn có thể dùng sd-legacy/stable-diffusion-v1-5.
# Hôm nay mình đổi qua stablediffusionapi/anything-v5 để tạo ảnh anime cho nó bớt nhàm chán nha!
pipeline = DiffusionPipeline.from_pretrained("stablediffusionapi/anything-v5",
                                             use_safetensors=True, safety_checker=None, requires_safety_checker=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline.to(device)
print(pipeline)

image = pipeline("girl with puppy ears").images[0]
image.show()  # Unconmment nếu chạy ở local

# Lưu ảnh
image.save("puppy_cartoon.png")


prompt = "girl with puppy ears"
steps = 30

pipeline.scheduler = EulerDiscreteScheduler.from_config(
    pipeline.scheduler.config)
# Sử dụng cùng một prompt với 3 scheduler khác nhau mới
image = pipeline(prompt, num_inference_steps=steps).images[0]
image.show()

pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
image = pipeline(prompt, num_inference_steps=steps).images[0]
image.show()

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
    pipeline.scheduler.config)
image = pipeline(prompt, num_inference_steps=steps).images[0]
image.show()

# Thay đổi height và width của ảnh
image = pipeline("cute puppy girl reading a book",
                 height=768, width=512).images[0]
image.show()

# dùng chung prompt với 3 giá trị guidance_scale khác nhau
main_prompt = "cute puppy girl reading a book"

image = pipeline(main_prompt, guidance_scale=1).images[0]
image.show()

image = pipeline(main_prompt, guidance_scale=7).images[0]
image.show()

image = pipeline(main_prompt, guidance_scale=40).images[0]
image.show()

# negative prompt
image = pipeline(
    prompt="cute puppy girl, detailed, 8k, best quality",
    negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy, low quality, worst quality",
).images[0]
image.show()


# Ảnh 1 giống ảnh 2 vì chung seed, ảnh 3 sẽ khác
prompt = "cute puppy girl with flower"

generator = torch.Generator(device).manual_seed(42)
image = pipeline(prompt, generator=generator).images[0]
image.show()

generator = torch.Generator(device).manual_seed(42)
image = pipeline(prompt, generator=generator).images[0]
image.show()

generator = torch.Generator(device).manual_seed(44)
image = pipeline(prompt, generator=generator).images[0]
image.show()

# Tổng kết, sử dụng các parameter khác nhau
image = pipeline(
    "puppy girl, close up, portrait",
    height=512,
    width=768,
    guidance_scale=6.5,
    num_inference_steps=24,
    negative_prompt="ugly, deformed, disfigured, low quality, worst quality",
    generator=torch.Generator(device=device).manual_seed(6969),
).images[0]
image.show()