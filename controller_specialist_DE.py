import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller

# load the best solution's weights
experiment_name = 'optimization_test_DE_247'
best_solution_file = experiment_name + '/best.txt'
best_solution_weights = np.loadtxt(best_solution_file)
n_hidden_neurons=10

# environment params
env = Environment(
    experiment_name=experiment_name,
    enemies=[7], 
    playermode="ai",
    player_controller=player_controller(n_hidden_neurons),
    enemymode="static",
    level=2,
    speed="normal",
    visuals=True
)

# play the best solution
fitness, player_life, enemy_life, time_elapsed = env.play(pcont=best_solution_weights)

# print results
print("Fitness:", fitness)
print("Player Life:", player_life)
print("Enemy Life:", enemy_life)
print("Time Elapsed:", time_elapsed)
