import os
import time
import random
import argparse
import pandas as pd

from diffusers import StableDiffusionPipeline
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import os

from image_generation import *
from fitness import *
from variation import *


parser = argparse.ArgumentParser(description="Script for running image generation with evolutionary algorithms.")
parser.add_argument("--target_prompt", type=str, default="two cars on the street", help="The target prompt for the image generation.")
parser.add_argument("--run_identifier", type=str, default="run_images", help="An identifier for the current run.")
parser.add_argument("--num_of_generations", type=int, default=15, help="Number of generations to evolve.")
parser.add_argument("--population_size", type=int, default=30, help="Size of the population for the evolutionary algorithm.")
parser.add_argument("--log_filename", type=str, default="log.csv", help="Name of the log file to store the results.")
parser.add_argument("--crossover_name", type=str, default="basic_latent_blend_crossover", help="The name of the crossover strategy to use.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the run.")
args = parser.parse_args()

target_prompt = args.target_prompt
run_identifier = args.run_identifier
num_of_generations = args.num_of_generations
population_size = args.population_size
log_filename = args.log_filename
crossover_name = args.crossover_name
seed = args.seed

random.seed(seed)

model_name = "rev-animated122" 
txt2img_pipeline = StableDiffusionPipeline.from_pretrained(model_name).to("cuda")
img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model_name).to("cuda")

actual_crossover = eval(crossover_name)

output_folder = "output_images/" + run_identifier
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

log_df = None
if os.path.exists(log_filename):
    log_df = pd.read_csv(log_filename)
else:
    log_df = pd.DataFrame(columns=['run_identifier', 'target_prompt', 'num_gen', 'pop_size', 'defined_crossover', 'generation', 'individual', 'image_path', 'fitness', 'comment'])

print("Generation: 0")
best_individual = None 
population = []    # Individuals are just tuples consisting of path and fitness
for i in range(0, population_size):
    current_filename = text_to_image_generation(txt2img_pipeline, target_prompt, f"{output_folder}/gen0_{time.time()}.png", random.randint(1000, 999999999), 1.0)
    current_individual = (current_filename, fitness_eval(current_filename, target_prompt))
    population.append(current_individual)
    log_df.loc[len(log_df)] = [run_identifier, target_prompt, num_of_generations, population_size, crossover_name, 0, '0_' + str(i), current_individual[0], current_individual[1], '-']
best_individual = max(population, key=lambda x: x[1])    # Find best individual of initial generation

for generation in range(1, num_of_generations + 1):
    population = tournament_selection(population)
    random.shuffle(population)
    print(f"Generation: {generation}")
    variated_population = []
    for j in range(0, len(population) - 1):
        blended_image = actual_crossover(txt2img_pipeline, population[j], population[j + 1], random.uniform(0.0, 1.0))
        current_filename = image_to_image_generation(img2img_pipeline, target_prompt, blended_image, f"{output_folder}/gen{generation}_{time.time()}.png", random.randint(1000, 999999999), random.uniform(0.6, 0.95))
        variated_individual = (current_filename, fitness_eval(current_filename, target_prompt))
        variated_population.append(variated_individual)
    variated_population.append(best_individual)    # Elitism (add best)
    population = variated_population
    best_individual = max(population, key=lambda x: x[1])    # Find new best indidivual
    for k, current_individual in enumerate(population):
        log_df.loc[len(log_df)] = [run_identifier, target_prompt, num_of_generations, population_size, crossover_name, generation, str(generation) + '_' + str(k), current_individual[0], current_individual[1], '-']

print(best_individual)
log_df.loc[len(log_df)] = [run_identifier, target_prompt, num_of_generations, population_size, crossover_name, -1, '-', best_individual[0], best_individual[1], 'best_image']
log_df.to_csv(log_filename, index=False)


