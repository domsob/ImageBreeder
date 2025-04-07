import torch
import random
import ImageReward as RM

image_reward_model = RM.load("ImageReward-v1.0")

def fitness_eval(filename, prompt):
    def image_reward(filename, prompt):
        try:
            return (image_reward_model.score(prompt, [filename]))
        except:
            return -999.0
    
    retry_counter = 0
    score = image_reward(filename, prompt)
    while score == -999.0:
        score = image_reward(filename, prompt)
        if retry_counter >= 25:
            break
        retry_counter += 1

    return score

def tournament_selection(pop, k=5):
    offspring = []
    pop_size = len(pop)
    while len(offspring) < pop_size:
        tournament_contestants = random.sample(pop, k)
        winner = max(tournament_contestants, key=lambda x: x[1])
        offspring.append(winner)
    return offspring
    

    
