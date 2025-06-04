'''
Animation for the double pole balancing problem.
'''

import os
import pickle

import matplotlib.animation as animation
import matplotlib.pyplot as plt

import examples.cart2pole.cart_two_pole_base as cart

fig = plt.figure(figsize=(8, 2.4))
ax = fig.add_subplot(1, 1, 1)

pole_length_1 = 1.0 / 2.0
pole_length_2 = 0.1 / 2.0
min_x, max_x = -2.5, 2.5

max_bal_steps = 1000

path = os.path.join(os.path.dirname(__file__), 'results_cart_0')
# path = 'results_cart_0'
gen = '499'

with open('{}/data_ind_gen{:0>4}.pkl'.format(path, gen), 'rb') as pkl:
    ind = pickle.load(pkl)

anim_cart = cart.CartAnimation(ax, ind, min_x, max_x,
                               pole_length_1=pole_length_1, pole_length_2=pole_length_2,
                               max_bal_steps=max_bal_steps)

ani = animation.FuncAnimation(fig, anim_cart.run_simulation_anim, interval=17,
                              frames=max_bal_steps, repeat=False)
# ani.save('{}/animation_gen{:0>4}.gif'.format(path,gen), writer='imagemagick')
# ani.save('{}/animation_gen{:0>4}.mp4'.format(path,gen), writer='ffmpeg')
plt.show()
