import pandas as pd
import matplotlib.pyplot as plt

path1 = 'optimization_test/results.txt'
path2 = 'optimization_test_DE/results.txt'
data1 = pd.read_csv(path1, sep=' ')
data2 = pd.read_csv(path2, sep=' ')

# columns
generation1 = data1['gen']
best_values1 = data1['best']
mean_values1 = data1['mean']

generation2 = data2['gen']
best_values2 = data2['best']
mean_values2 = data2['mean']


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

ax1.plot(generation1, best_values1, label='Best', marker='o')
ax1.plot(generation1, mean_values1, label='Mean', marker='x')
ax1.set_ylabel('Fitness Value (Path 1)')
ax1.set_title('Demo algorithm (Genetic Algorithm)')
ax1.legend()
ax1.grid(True)
ax1.set_ylim(0, 100)

ax2.plot(generation2, best_values2, label='Best', marker='o')
ax2.plot(generation2, mean_values2, label='Mean', marker='x')
ax2.set_xlabel('Generation')
ax2.set_ylabel('Fitness Value (Path 2)')
ax2.set_title('Test algorithm (Differential Evolution)')
ax2.legend()
ax2.grid(True)
ax2.set_ylim(0, 100)

plt.tight_layout()
plt.show()

