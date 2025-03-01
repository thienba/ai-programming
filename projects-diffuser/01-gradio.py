import gradio as gr
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    EulerDiscreteScheduler,
)
import torch
from typing import Optional, Tuple, Dict, Any
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    name: str
    use_safetensors: bool = True
    safety_checker: Optional[Any] = None
    torch_dtype: torch.dtype = torch.float32
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    num_inference_steps: int = 30

    def get_scheduler(self) -> EulerDiscreteScheduler:
        return EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000
        )

class StableDiffusionGenerator:
    def __init__(self, initial_model: str = "runwayml/stable-diffusion-v1-5"):
        self.device = self._get_optimal_device()
        self.config = ModelConfig(name=initial_model)
        self.pipe = self._initialize_pipeline()
        
    def _get_optimal_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _initialize_pipeline(self) -> Any:
        try:
            logger.info(f"Initializing model {self.config.name} on {self.device}")
            PipelineClass = self._get_pipeline_for_model(self.config.name)
            
            if self.device == "cuda":
                self.config.torch_dtype = torch.float16
            
            pipe = PipelineClass.from_pretrained(
                self.config.name,
                use_safetensors=self.config.use_safetensors,
                safety_checker=self.config.safety_checker,
                torch_dtype=self.config.torch_dtype,
                scheduler=self.config.get_scheduler()
            )
            
            if self.config.enable_attention_slicing:
                pipe.enable_attention_slicing()
            if self.config.enable_vae_slicing:
                pipe.enable_vae_slicing()
                
            return pipe.to(self.device)
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {str(e)}")
            raise

    def _get_pipeline_for_model(self, model_name: str) -> Any:
        return StableDiffusionXLPipeline if "xl" in model_name.lower() else StableDiffusionPipeline

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str,
        model: str,
        seed: int,
        num_inference_steps: int,
        guidance_scale: float,
        width: int,
        height: int
    ) -> Any:
        try:
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            else:
                generator = None
            
            return self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator
            ).images[0]
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            raise

    def load_model(self, model_name: str) -> Tuple[str, gr.update]:
        try:
            self.config.name = model_name
            self.pipe = self._initialize_pipeline()
            return "Model loaded successfully!", gr.update(interactive=True)
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            logger.error(error_msg)
            return error_msg, gr.update(interactive=True)

def create_ui(generator: StableDiffusionGenerator) -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("# Stable Diffusion Demo")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt")
                negative_prompt = gr.Textbox(label="Negative Prompt")
                model = gr.Dropdown(
                    choices=[
                        "stablediffusionapi/anything-v5",
                        "runwayml/stable-diffusion-v1-5",
                        "CompVis/stable-diffusion-v1-4",
                        "stabilityai/stable-diffusion-2-1",
                    ],
                    value="stablediffusionapi/anything-v5",
                    label="Model"
                )
                with gr.Row():
                    width = gr.Slider(minimum=128, maximum=1024, step=64, value=512, label="Width")
                    height = gr.Slider(minimum=128, maximum=1024, step=64, value=512, label="Height")
                seed = gr.Slider(minimum=0, maximum=2**32 - 1, step=1, value=42, label="Seed")
                num_inference_steps = gr.Slider(minimum=1, maximum=100, step=1, value=30, label="Steps")
                guidance_scale = gr.Slider(minimum=0, maximum=20, step=0.5, value=7.5, label="Guidance Scale")
            
            with gr.Column():
                output_image = gr.Image(label="Output Image")
                status = gr.Textbox(label="Status")
        
        generate_button = gr.Button("Generate Image")
        
        generate_button.click(
            generator.generate_image,
            inputs=[prompt, negative_prompt, model, seed, num_inference_steps, guidance_scale, width, height],
            outputs=output_image
        )
        
        model.change(
            generator.load_model,
            inputs=[model],
            outputs=[status, generate_button]
        )
        
        return demo

if __name__ == "__main__":
    try:
        generator = StableDiffusionGenerator()
        demo = create_ui(generator)
        demo.launch()
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")