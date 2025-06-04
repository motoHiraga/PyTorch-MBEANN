'''
Animation for the Gym problem.
'''

import os
import pickle
import time

import gymnasium as gym
import numpy as np


def isDiscreteActions(env):
    return 'Discrete' in str(type(env.action_space))


# --- Gym settings. --- #
# Only supports environments with the following state and action spaces:
# env.observation_space - Box(X,)
# env.action_space      - Box(X,) or Discrete(X)
envName = 'HalfCheetah-v5'

# Episode length should be longer than the termination condition defined in the gym environment.
episode_length = 100000

# Load MBEANN individual data
path = os.path.join(os.path.dirname(__file__), 'results_gym_0')
gen = '99'

with open('{}/data_ind_gen{:0>4}.pkl'.format(path, gen), 'rb') as pkl:
    ind = pickle.load(pkl)

# Make gym environment.
env = gym.make(envName, render_mode='human')

total_reward = 0
randomSeed = int(time.time())
observation, info = env.reset(seed=randomSeed)

for t in range(episode_length):

    env.render()

    action = ind.calculateNetwork(observation)

    if isDiscreteActions(env):
        action = np.argmax(action)
    else:
        action = action * (env.action_space.high - env.action_space.low) + env.action_space.low

    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    if terminated or truncated:
        print("Episode finished after {} timesteps with getting reward {}".format(t + 1, total_reward))
        break

env.close()
