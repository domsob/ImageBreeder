from diffusers import StableDiffusionPipeline
from PIL import Image
import random
import torch
import torchvision.transforms as transforms
import numpy as np
import os

def vae_encode(pipeline, image_path):
    image = Image.open(image_path).convert("RGB")
    vae = pipeline.vae

    preprocess = transforms.Compose([
        transforms.Resize((512, 512)), 
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) 
    ])
    image_tensor = preprocess(image).unsqueeze(0).to(vae.device)

    with torch.no_grad():
        latents = vae.encode(image_tensor).latent_dist.sample() 
        latents = latents * vae.config.scaling_factor 
    
    return latents

def vae_decode(pipeline, latents):
    vae = pipeline.vae

    with torch.no_grad():
        decoded_output = vae.decode(latents / vae.config.scaling_factor)  
        decoded_tensor = decoded_output.sample 

    decoded_tensor = (decoded_tensor.clamp(-1, 1) + 1) / 2 
    decoded_tensor = decoded_tensor.squeeze().permute(1, 2, 0).cpu().numpy()  

    reconstructed_image = (decoded_tensor * 255).astype("uint8")
    reconstructed_pil = Image.fromarray(reconstructed_image)
    #reconstructed_pil.save("output.png")

    return reconstructed_pil

def basic_latent_blend_crossover(pipeline, individ1, individ2, blend_alpha=0.5):
    latent1 = vae_encode(pipeline, individ1[0])
    latent2 = vae_encode(pipeline, individ2[0])

    blended_latent = (1 - blend_alpha) * latent1 + blend_alpha * latent2

    return vae_decode(pipeline, blended_latent)

def fitness_latent_blend_crossover(pipeline, individ1, individ2, _):
    score1 = individ1[1] if individ1[1] > 0 else 0
    score2 = individ2[1] if individ2[1] > 0 else 0
    if score1 + score2 == 0:
        blend_alpha = 0.5
    else:
        blend_alpha = score2 / (score1 + score2)
    return basic_latent_blend_crossover(pipeline, individ1, individ2, blend_alpha)

def pixel_blend_crossover(_, individ1, individ2, crossover_point=0.5):
    image1 = Image.open(individ1[0]).convert("RGB")
    image2 = Image.open(individ2[0]).convert("RGB")

    image1 = image1.resize((512, 512))
    image2 = image2.resize((512, 512))

    blended_image = Image.blend(image1, image2, alpha=crossover_point)

    return blended_image

def fitness_pixel_blend_crossover(pipeline, individ1, individ2, _):
    score1 = individ1[1] if individ1[1] > 0 else 0
    score2 = individ2[1] if individ2[1] > 0 else 0
    if score1 + score2 == 0:
        blend_alpha = 0.5
    else:
        blend_alpha = score2 / (score1 + score2)
    return pixel_blend_crossover(pipeline, individ1, individ2, blend_alpha)

def channel_cut_crossover(pipeline, individ1, individ2, crossover_point=0.5):
    latent1 = vae_encode(pipeline, individ1[0])
    latent2 = vae_encode(pipeline, individ2[0])
    
    if crossover_point >= 0.75:
        channel_cut = 3
    elif crossover_point <= 0.25:
        channel_cut = 1
    else:
        channel_cut = 2

    offspring1 = torch.cat((latent1[:, :channel_cut], latent2[:, channel_cut:]), dim=1)
    offspring2 = torch.cat((latent2[:, :channel_cut], latent1[:, channel_cut:]), dim=1)

    return vae_decode(pipeline, random.choice([offspring1, offspring2]))

def horizontal_cut_crossover(pipeline, individ1, individ2, crossover_point=0.5):
    latent1 = vae_encode(pipeline, individ1[0])
    latent2 = vae_encode(pipeline, individ2[0])
    
    cut_point = int(crossover_point*64) 

    offspring1 = torch.cat((latent1[:, :, :cut_point, :], latent2[:, :, cut_point:, :]), dim=2)
    offspring2 = torch.cat((latent2[:, :, :cut_point, :], latent1[:, :, cut_point:, :]), dim=2)

    return vae_decode(pipeline, random.choice([offspring1, offspring2]))

def vertical_cut_crossover(pipeline, individ1, individ2, crossover_point=0.5):
    latent1 = vae_encode(pipeline, individ1[0])
    latent2 = vae_encode(pipeline, individ2[0])
    
    cut_point = int(crossover_point*64) 

    offspring1 = torch.cat((latent1[:, :, :, :cut_point], latent2[:, :, :, cut_point:]), dim=3)
    offspring2 = torch.cat((latent2[:, :, :, :cut_point], latent1[:, :, :, cut_point:]), dim=3)

    return vae_decode(pipeline, random.choice([offspring1, offspring2]))

def random_cut_crossover(pipeline, individ1, individ2, crossover_point=0.5):
    crossover_methods = [channel_cut_crossover, horizontal_cut_crossover, vertical_cut_crossover]
    used_crossover = random.choice(crossover_methods)

    return used_crossover(pipeline, individ1, individ2, crossover_point)

def any_crossover(pipeline, individ1, individ2, crossover_point=0.5):
    crossover_methods = [basic_latent_blend_crossover, fitness_latent_blend_crossover, pixel_blend_crossover, 
        fitness_pixel_blend_crossover, channel_cut_crossover, horizontal_cut_crossover, vertical_cut_crossover]
    used_crossover = random.choice(crossover_methods)

    return used_crossover(pipeline, individ1, individ2, crossover_point)

def only_mutation(pipeline, individ1, individ2, _):
    latent1 = vae_encode(pipeline, individ1[0])
    return vae_decode(pipeline, latent1)

