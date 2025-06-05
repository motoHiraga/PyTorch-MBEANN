'''
Example of MBEANN solving the Gym problem.

Gymnasium: https://gymnasium.farama.org
'''

import os
import pickle
import random
import time

import gymnasium as gym
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from examples.gym.settings import SettingsEA, SettingsMBEANN
from pytorch_mbeann.base import Individual, ToolboxMBEANN
from pytorch_mbeann.visualize import visualizeIndividual

# --- Gym settings. --- #
# Only supports environments with the following state and action spaces:
# env.observation_space - Box(X,)
# env.action_space      - Box(X,) or Discrete(X)
envName = 'HalfCheetah-v5'

# Make gym environment.
env = gym.make(envName)


def isDiscreteActions(env):
    return 'Discrete' in str(type(env.action_space))


def evaluateIndividual(ind):
    total_reward = 0

    # Number of evaluations per individual.
    episode_per_ind = 1

    # Episode length should be longer than the termination condition defined in the gym environment.
    episode_length = 100000

    for i_episode in range(episode_per_ind):

        observation, info = env.reset()

        for t in range(episode_length):

            action = ind.calculateNetwork(observation)

            if isDiscreteActions(env):
                action = np.argmax(action)
            else:
                action = action * (env.action_space.high - env.action_space.low) + env.action_space.low

            observation, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            if terminated or truncated:
                break

    fitness = total_reward / episode_per_ind
    env.close()
    return fitness,


if __name__ == '__main__':

    start_time = time.time()

    # Parallel evaluation settings.
    multiProcessLib = 'multiprocessing'  # 'mpi4py' or 'multiprocessing'

    if multiProcessLib == 'mpi4py':
        from mpi4py import MPI
        from mpi4py.futures import MPIPoolExecutor
        pool = MPIPoolExecutor()

    elif multiProcessLib == 'multiprocessing':
        import torch.multiprocessing as multiprocessing
        if multiprocessing.get_start_method() == 'fork':
            multiprocessing.set_start_method('spawn', force=True)
        processCount = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=processCount)
    else:
        processCount = 1

    # Get input size from the gym environment.
    if SettingsMBEANN.inSize is None:
        SettingsMBEANN.inSize = env.observation_space.shape[0]

    # Get output size from the gym environment.
    if SettingsMBEANN.outSize is None:
        if isDiscreteActions(env):
            SettingsMBEANN.outSize = env.action_space.n
        else:
            SettingsMBEANN.outSize = env.action_space.shape[0]

    # Evolutionary algorithm settings.
    popSize = SettingsEA.popSize
    maxGeneration = SettingsEA.maxGeneration
    isMaximizingFit = SettingsEA.isMaximizingFit
    eliteSize = SettingsEA.eliteSize
    tournamentSize = SettingsEA.tournamentSize
    tournamentBestN = SettingsEA.tournamentBestN

    randomSeed = 0  # int(time.time())
    random.seed(randomSeed)
    st = random.getstate()

    # Changed 'env.seed(seed)' to 'env.reset(seed=seed)'.
    # Might lose reproducibility at 'env.reset()' in 'evaluateIndividual'.
    env.reset(seed=randomSeed)

    # Path to save data.
    data_dir = os.path.join(os.path.dirname(__file__), f'results_gym_{randomSeed}')
    os.makedirs(data_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=data_dir)

    with open(f'{data_dir}/random_state.pkl', mode='wb') as out_pkl:
        pickle.dump(st, out_pkl)

    pop = [Individual(inputSize=SettingsMBEANN.inSize,
                      outputSize=SettingsMBEANN.outSize,
                      hiddenSize=SettingsMBEANN.hidSize,
                      initialConnection=SettingsMBEANN.initialConnection,
                      maxWeight=SettingsMBEANN.maxWeight,
                      minWeight=SettingsMBEANN.minWeight,
                      initialWeightType=SettingsMBEANN.initialWeightType,
                      initialWeightMean=SettingsMBEANN.initialWeighMean,
                      initialWeightScale=SettingsMBEANN.initialWeightScale,
                      maxBias=SettingsMBEANN.maxBias,
                      minBias=SettingsMBEANN.minBias,
                      initialBiasType=SettingsMBEANN.initialBiasType,
                      initialBiasMean=SettingsMBEANN.initialBiasMean,
                      initialBiasScale=SettingsMBEANN.initialBiasScale,
                      maxStrategy=SettingsMBEANN.maxStrategy,
                      minStrategy=SettingsMBEANN.minStrategy,
                      initialStrategy=SettingsMBEANN.initialStrategy,
                      isRecurrent=SettingsMBEANN.isRecurrent,
                      activationFunc=SettingsMBEANN.activationFunc,
                      addNodeBias=SettingsMBEANN.actFuncBias,
                      addNodeGain=SettingsMBEANN.actFuncGain)
           for i in range(popSize)]
    tools = ToolboxMBEANN(p_addNode=SettingsMBEANN.p_addNode,
                          p_addLink=SettingsMBEANN.p_addLink,
                          p_weight=SettingsMBEANN.p_weight,
                          p_bias=SettingsMBEANN.p_bias,
                          mutWeightType=SettingsMBEANN.weightMutationType,
                          mutWeightScale=SettingsMBEANN.weightMutationScale,
                          mutBiasType=SettingsMBEANN.biasMutationType,
                          mutBiasScale=SettingsMBEANN.biasMutationScale,
                          mutationProbCtl=SettingsMBEANN.mutationProbCtl,
                          addNodeWeight=SettingsMBEANN.addNodeWeightValue)

    log_stats = ['Gen', 'Mean', 'Std', 'Max', 'Min']
    with open(f'{data_dir}/log_stats.pkl', mode='wb') as out_pkl:
        pickle.dump(log_stats, out_pkl)

    for gen in range(maxGeneration):
        print('------')
        print(f'Gen {gen}')

        if multiProcessLib == 'mpi4py':
            fitnessValues = list(pool.map(evaluateIndividual, pop))
        else:
            if processCount > 1:
                fitnessValues = pool.map(evaluateIndividual, pop)
            else:
                fitnessValues = []
                for ind in pop:
                    fitnessValues += [evaluateIndividual(ind)]

        for ind, fit in zip(pop, fitnessValues):
            ind.fitness = fit[0]

        log_stats = [gen, np.mean(fitnessValues), np.std(fitnessValues),
                     np.max(fitnessValues), np.min(fitnessValues)]

        with open(f'{data_dir}/log_stats.pkl', mode='ab') as out_pkl:
            pickle.dump(log_stats, out_pkl)

        print('Mean: ' + str(np.mean(fitnessValues)) +
              '\tStd: ' + str(np.std(fitnessValues)) +
              '\tMax: ' + str(np.max(fitnessValues)) +
              '\tMin: ' + str(np.min(fitnessValues)), flush=True)

        writer.add_scalar('Fitness/Mean', np.mean(fitnessValues), gen)
        writer.add_scalar('Fitness/Std', np.std(fitnessValues), gen)
        writer.add_scalar('Fitness/Max', np.max(fitnessValues), gen)
        writer.add_scalar('Fitness/Min', np.min(fitnessValues), gen)

        # Save the best individual.
        with open(f'{data_dir}/data_ind_gen{gen:0>4}.pkl', mode='wb') as out_pkl:
            pop.sort(key=lambda ind: ind.fitness, reverse=isMaximizingFit)
            pickle.dump(pop[0], out_pkl)

        visualizeIndividual(pop[0], f'{data_dir}/mbeann_ind_gen{gen:0>4}.pdf')

        writer.add_scalar('Complexity/BestIndNumNodes', pop[0].getNodeAndLinkNum()[0], gen)
        writer.add_scalar('Complexity/BestIndNumLinks', pop[0].getNodeAndLinkNum()[1], gen)

        tools.selectionSettings(pop, popSize, isMaximizingFit, eliteSize)

        if eliteSize > 0:
            elite = tools.preserveElite()

        pop = tools.selectionTournament(tournamentSize, tournamentBestN)

        for i, ind in enumerate(pop):
            tools.mutateWeightAndBiasValue(ind)
            # tools.mutateWeightValue(ind)
            # tools.mutateBiasValue(ind)
            tools.mutateAddNode(ind, prob_lower_bound=0.01)
            tools.mutateAddLink(ind, prob_lower_bound=0.1)

        if eliteSize > 0:
            pop = elite + pop

        elapsed_time = time.time() - start_time
        writer.add_scalar('Performance/TimePerGen', elapsed_time, gen)
        start_time = time.time()

        writer.flush()

    writer.close()

    if multiProcessLib == 'multiprocessing':
        pool.close()
        pool.join()

    if multiProcessLib == 'mpi4py':
        pool.shutdown(wait=False)
        # MPI.COMM_WORLD.Abort(1)  # Use only if needed on systems prone to hanging
