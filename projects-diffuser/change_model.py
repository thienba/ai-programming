import requests

url = "http://127.0.0.1:7860"
option_payload = {
    "sd_model_checkpoint": "realisticVisionV60B1_v51HyperVAE.safetensors",
}
response = requests.post(url=f'{url}/sdapi/v1/options', json=option_payload)
print(response.json())
