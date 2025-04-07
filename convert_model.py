from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
import torch

# Path to safetensors file
safetensors_path = ".../rev-animated-v1-2-2.safetensors"

pipeline = StableDiffusionPipeline.from_single_file(safetensors_path, torch_dtype=torch.float16)
pipeline.save_pretrained(".../rev-animated122")
