import services
import io
import base64

from models import ImageRequest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to Stable Diffusers API"}

@app.post("/api/v1/generate/")
async def generate_image(imgRequest: ImageRequest):
    image = await services.generate_image(imgRequest=imgRequest)

    memory_stream = io.BytesIO()
    image.save(memory_stream, format="PNG")
    memory_stream.seek(0)
    return StreamingResponse(memory_stream, media_type="image/png")

@app.post("/api/v1/generatebase64/")
async def generate_base64_image(imgRequest: ImageRequest):
    image = await services.generate_image(imgRequest=imgRequest)

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return {
        "image": f"data:image/png;base64,{img_base64}"
    }