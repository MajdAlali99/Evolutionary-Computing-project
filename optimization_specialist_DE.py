import sys
from evoman.environment import Environment
from demo_controller import player_controller
import time
import numpy as np
import os

# parameters
dom_l = -1
dom_u = 1
npop = 1000
gens = 50
mutation = 0.5
last_best = 0
fim = 0
CR = 0.2
strategy = 'DE/best/1'
# strategy = 'DE/current-to-best/1'

experiment_name = 'DE_3_gen_50_CR_0.2_mutation_0.5_pop_1000'

def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f

# DE-specific functions
def mutation_operation(pop, best_idx, F, current_idx):

    if strategy == 'DE/best/1':
        idxs = list(range(npop))
        idxs.remove(best_idx)

        # best individual
        a = pop[best_idx]

        # two random individuals
        b, c = np.random.choice(idxs, 2, replace=False)

        # Mutate
        mutant = a + F * (pop[b] - pop[c])
        mutant = np.clip(mutant, dom_l, dom_u)
    
    elif strategy == 'DE/current-to-best/1':
        idxs = list(range(npop))
        idxs.remove(current_idx)

        # best individual
        best = pop[best_idx]

        # random individual
        a = np.random.choice(idxs, 1)[0]

        # Mutate
        mutant = pop[current_idx] + F * (best - pop[current_idx]) + F * (pop[a] - pop[current_idx])
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

def main():

    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # environment params
    env = Environment(experiment_name=experiment_name,
                    enemies=[8],
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
            trial = crossover_operation(pop[j], mutant, CR=CR)

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
