import base64
import requests

URL = "http://127.0.0.1:7860"


def base64_to_image(base64_string, save_path='output_image.png'):
    with open(save_path, 'wb') as f:
        f.write(base64.b64decode(base64_string))


def text_to_image():
    print("Starting Inference")
    payload = {
        "prompt": "elf, neon hair, neon eyes, ponytail hairstyle, black leather jacket",
        "negative_prompt": "worst quality, low quality, watermark, text, error, blurry, jpeg artifacts, cropped, jpeg artifacts, signature, watermark, username, artist name, bad anatomy",
        "steps": 25,
        "cfg_scale": 7.5,
        "width": 512,
        "height": 512,
    }

    response = requests.post(f"{URL}/sdapi/v1/txt2img", json=payload)
    resp_json = response.json()
    print("Inference Completed")
    for i, img in enumerate(resp_json['images']):
        print(f"Saving image output_image_{i}.png")
        base64_to_image(img, f"output_image_{i}.png")


if __name__ == "__main__":
    text_to_image()


# import requests
# url = "http://127.0.0.1:7860"

# option_payload = {
#     "sd_model_checkpoint": "juggernaut_reborn",
# }

# response = requests.post(url=f'{url}/sdapi/v1/options', json=option_payload)

# print("Status Code:", response.status_code)
# if response.status_code == 200:
#     print("Options Set Successfully")
# else:
#     print("Failed to set options")