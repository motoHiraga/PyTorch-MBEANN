'''
Example for visualizing fitness progression during the evolution process.
'''

import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

path = os.path.join(os.path.dirname(__file__), 'examples/xor/results_xor_0')
# path = os.path.join(os.path.dirname(__file__), 'examples/cart2pole/results_cart_0')

logs = []
with open(f'{path}/log_stats.pkl', 'rb') as pkl:
    while True:
        try:
            o = pickle.load(pkl)
        except EOFError:
            break
        logs.append(o)

df = pd.DataFrame(logs[1:], columns=logs[0])

print(df)

fig = plt.figure()
ax = fig.add_subplot(111)

# ax = df.plot(x='Gen', colormap='viridis', linewidth=3.0)

plt_mean = plt.plot(df['Gen'], df['Mean'],
                    linewidth=2.0,
                    color='cornflowerblue',
                    label='mean')

plt_std = plt.fill_between(df['Gen'],
                           df['Mean'] - df['Std'],
                           df['Mean'] + df['Std'],
                           color='cornflowerblue',
                           alpha=0.4,
                           linewidth=0,
                           label='standard deviation \nof mean')

plt_max = plt.plot(df['Gen'], df['Max'],
                   linewidth=2.0,
                   linestyle='--',
                   color='g',
                   label='max')

plt_min = plt.plot(df['Gen'], df['Min'],
                   linewidth=2.0,
                   linestyle=':',
                   color='y',
                   label='min')

ax.tick_params(labelsize=14)
ax.set_xlabel(r'Generation', fontsize=18)
ax.set_ylabel(r'Fitness Value', fontsize=18)
ax.legend(fontsize=12)
# ax.legend(loc='lower right', fontsize=12)


plt.tight_layout()
plt.savefig(f'{path}/mbeann_fitness.pdf')
plt.show()
