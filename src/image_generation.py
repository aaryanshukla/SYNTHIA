import torch
from diffusers import StableDiffusionPipeline
from config import DEVICE, MODEL_CACHE_DIR

class ImageGenerator:
    def __init__(self, model_name="runwayml/stable-diffusion-v1-5"):
        print(f"[ImageGenerator] Loading Stable Diffusion model '{model_name}' on device: {DEVICE}")
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            revision="fp16",
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE_DIR
        )
        self.pipeline.to(DEVICE)
        
    def generate_image(self, prompt, guidance_scale=7.5, num_inference_steps=50):
        with torch.no_grad():
            image = self.pipeline(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
        return image
