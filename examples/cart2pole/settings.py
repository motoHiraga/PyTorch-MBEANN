'''
MBEANN settings for the double pole balancing problem.
'''


class SettingsEA:

    # --- Evolutionary algorithm settings. --- #
    popSize = 1000
    maxGeneration = 500  # 0 to (maxGeneration - 1)
    isMaximizingFit = True
    eliteSize = 0
    tournamentSize = 20
    tournamentBestN = 1  # Select best N individuals from each tournament.


class SettingsMBEANN:

    # --- Neural network settings. --- #
    inSize = 3
    outSize = 1
    hidSize = 0

    isRecurrent = True

    # initialConnection: (0.0, 1.0]
    # 1.0 for initialization using the fully-connected topology.
    # Between 0.0 to 1.0 for partial connections.
    initialConnection = 1.0

    # initialWeightType: 'uniform', 'gaussian', or 'cauchy'
    # uniform  - Uniform random numbers between minWeight to maxWeight.
    # gaussian - Sampled from Gaussian distribution with 
    #            random.normalvariate(mu=initialWeighMean, sigma=initialWeightScale). 
    # cauchy   - Sampled from Cauchy distribution with the location parameter of initialWeighMean
    #            and the scale parameter of initialWeightScale.
    # Weights out of the range [minWeight, maxWeight] are clipped.
    initialWeightType = 'gaussian'
    initialWeighMean = 0.0
    initialWeightScale = 0.5
    maxWeight = 5.0
    minWeight = -5.0

    # Bias settings.
    initialBiasType = 'gaussian'
    initialBiasMean = 0.0
    initialBiasScale = 0.5
    maxBias = 5.0
    minBias = -5.0

    # Strategy settings for "sa_one."
    initialStrategy = 0.5
    maxStrategy = 5.0
    minStrategy = 0.01

    # --- Mutation settings. --- #
    # Probability of mutations.
    p_addNode = 0.03
    p_addLink = 0.3
    p_weight = 1.0
    p_bias = 1.0

    # Controls mutation probabilities for add-node and add-connection mutations.
    # mutationProbCtl: 'operon' or 'network' (default: 'operon')
    # operon  - Mutations are applied to each operon with probabilities of p_addNode and p_addLink.
    # network - p_addNode and p_addLink are normalized with the number of operons.
    #           Lower bounds can be set by the argument in mutateAddNode and mutateAddLink.
    #           See [M. Hiraga, et al., 2024].
    mutationProbCtl = 'operon'  # 'operon' or 'network'

    # Settings for weight and bias mutations.
    # MutationType: 'uniform', 'gaussian', or 'cauchy'
    # uniform  - Replace the weight or bias value with the value sampled from
    #            the uniform random distribution between minWeight to maxWeight.
    # gaussian - Add the value sampled from Gaussian distribution with 
    #            random.normalvariate(mu=o, sigma=MutationScale). 
    # cauchy   - Add the value sampled from Cauchy distribution with the location parameter of 0
    #            and the scale parameter of MutationScale.
    # sa_one   - Self-adaptive mutation using uncorrelated mutation with one step size.
    #            See [A.E. Eiben and J.E. Smith, 2015; M. Hiraga, et al., 2024].
    #            Both weight and bias should be set to "sa_one."
    #            "MutationScale" is not used in this mutation. 
    # Values out of the range are clipped.
    weightMutationType = 'gaussian'
    weightMutationScale = 0.05
    biasMutationType = 'gaussian'
    biasMutationScale = 0.05

    # --- Activation function settings. --- #
    activationFunc = 'sigmoid'  # 'sigmoid' or 'tanh'
    addNodeWeightValue = 1.0

    # Recommended settings for 'sigmoid':
    actFuncBias = 0.5 * addNodeWeightValue
    actFuncGain = 4.629740 / addNodeWeightValue

    # Recommended settings for 'tanh':
    # actFuncBias = 0.0
    # actFuncGain = 1.157435 / addNodeWeightValue
