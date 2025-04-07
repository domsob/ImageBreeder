from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
import torch
from PIL import Image


def image_to_image_generation(pipeline, prompt, input_image, save_to, rnd_seed, diffusion_strenth=0.7):
    generator = torch.Generator("cuda").manual_seed(rnd_seed)
    generated_image = pipeline(prompt=prompt, image=input_image, num_inference_steps=30, guidance_scale=8.0, strength=diffusion_strenth, output_type="pil", generator=generator).images[0]
    generated_image.save(save_to)
    return save_to

def text_to_image_generation(pipeline,  prompt, save_to, rnd_seed, diffusion_strenth=1.0):
    generator = torch.Generator("cuda").manual_seed(rnd_seed)
    generated_image = pipeline(prompt=prompt, num_inference_steps=30, guidance_scale=8.0, strength=diffusion_strenth, output_type="pil", generator=generator).images[0]
    generated_image.save(save_to)
    return save_to
