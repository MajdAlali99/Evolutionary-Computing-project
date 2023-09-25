import sys
from evoman.environment import Environment
from demo_controller import player_controller
import time
import numpy as np
import os
import math


# parameters
dom_l = -1
dom_u = 1
npop = 100
gens = 30
mutation = 0.2
last_best = 0
fim = 0
max_similarity = 0.9

experiment_name = 'similarity_test_DE_sim_09'

def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f

# DE-specific functions
def mutation_operation(pop, best_idx, F):
    idxs = list(range(npop))
    idxs.remove(best_idx)

    # three random individuals
    a, b, c = np.random.choice(idxs, 3, replace=False)

    # Mutate
    mutant = pop[a] + F * (pop[b] - pop[c])
    mutant = np.clip(mutant, dom_l, dom_u)

    return mutant

def crossover_operation(target, mutant, CR):
    trial = np.copy(target)

    # random index for crossover
    idx = np.random.randint(0, len(target))

    # crossover
    for i in range(len(target)):
        if i == idx or np.random.rand() < CR:
            trial[i] = mutant[i]

    return trial

def evaluate(env, x):
    return np.array([simulation(env, ind) for ind in x])

def cosine_similarity(arr_1, arr_2):

    dot = np.dot(arr_1, arr_2)
    norm_arr_1 = np.linalg.norm(arr_1)
    norm_arr_2 = np.linalg.norm(arr_2)

    if norm_arr_1==0 or norm_arr_1==0:
        return 0.0

    similarity = dot / (norm_arr_1*norm_arr_2)

    return similarity

def can_mate(arr_1, arr_2, similarity_coef):
    sim = cosine_similarity(arr_1, arr_2)

    if abs(sim) < similarity_coef:
        return True
    else:
        return False


def main():

    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # environment params
    env = Environment(experiment_name=experiment_name,
                    enemies=[2],
                    multiplemode="no",
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons),
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)

    # Number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

    # Initialize population
    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    fit_pop = evaluate(env, pop)
    best_idx = np.argmax(fit_pop)
    best_sol = fit_pop[best_idx]

    # Log initialization
    file_aux = open(experiment_name+'/results.txt', 'a')
    file_aux.write('\n\ngen best mean std')
    file_aux.close()

    ini = time.time()  # sets time marker

    for i in range(gens):
        new_pop = np.zeros_like(pop)

        for j in range(npop):
            mutant = mutation_operation(pop, best_idx, mutation)

            if not can_mate(mutant, pop[j], max_similarity):
                continue

            trial = crossover_operation(pop[j], mutant, 0.7)

            trial_fitness = evaluate(env, [trial])[0]

            if trial_fitness > fit_pop[j]:
                pop[j] = trial
                fit_pop[j] = trial_fitness

                if trial_fitness > best_sol:
                    best_sol = trial_fitness
                    best_idx = j

            new_pop[j] = pop[j]

        pop = new_pop

        # Save logs
        mean = np.mean(fit_pop)
        std = np.std(fit_pop)
        file_aux = open(experiment_name+'/results.txt', 'a')
        print('\n GENERATION '+str(i)+' '+str(round(fit_pop[best_idx], 6))+' '+str(round(mean, 6))+' '+str(round(std, 6)))
        file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best_idx], 6))+' '+str(round(mean, 6))+' '+str(round(std, 6)))
        file_aux.close()

        # Save generation number
        file_aux = open(experiment_name+'/gen.txt', 'w')
        file_aux.write(str(i))
        file_aux.close()

        # Save file with the best solution
        np.savetxt(experiment_name+'/best.txt', pop[best_idx])

    fim = time.time() # prints total execution time for experiment
    print('\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
    print('\nExecution time: '+str(round((fim-ini)))+' seconds \n')

    file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
    file.close()

    env.state_to_log() # checks environment state

if __name__ == '__main__':
    main()