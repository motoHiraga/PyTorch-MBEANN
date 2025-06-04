'''
Main tools for MBEANN.
PyTorch-MBEANN (Author: Motoaki Hiraga)
'''

import copy
import itertools
import math
import torch
import random

import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Node:
    def __init__(self, id, type, bias=None):
        self.id = id
        self.type = type
        self.bias = bias
        self.value = 0.0


class Link:
    def __init__(self, id, fromNodeID, toNodeID, weight):
        self.id = id
        self.fromNodeID = fromNodeID
        self.toNodeID = toNodeID
        self.weight = weight


class Operon:
    def __init__(self, id, nodeList, linkList):
        self.id = id
        self.nodeList = nodeList
        self.linkList = linkList

    def setDisabledLinkList(self, linkList):
        # List of available links for mutateAddLink.
        self.disabledLinkList = linkList

    def addLinkToDisabledLinkList(self, fromNodeID, toNodeID):
        isExists = (fromNodeID, toNodeID) in self.disabledLinkList
        if isExists == True:
            # This warning might appear when using "isRecurrent = False".
            # mutateAddNode: Trying to delete a selected link again.
            # print(f"WARNING: {fromNodeID} to {toNodeID} connection already exists in the disabledLinkList of operon {self.id}")
            pass
        self.disabledLinkList += [(fromNodeID, toNodeID)]

    def deleteLinkFromDisabledLinkList(self, fromNodeID, toNodeID):
        isExists = (fromNodeID, toNodeID) in self.disabledLinkList
        if isExists == False:
            print(
                f"WARNING: Cannot find {fromNodeID} to {toNodeID} connection in the disabledLinkList of operon {self.id}")
        newDisabledLinkList = [i for i in self.disabledLinkList if i != (fromNodeID, toNodeID)]
        self.disabledLinkList = newDisabledLinkList


class Individual:
    # TODO: planning to use "configparser" to make the settings clean and organized.
    def __init__(self, inputSize, outputSize, hiddenSize, initialConnection,
                 maxWeight, minWeight, initialWeightType, initialWeightMean, initialWeightScale,
                 maxBias, minBias, initialBiasType, initialBiasMean, initialBiasScale,
                 maxStrategy, minStrategy, initialStrategy,
                 isRecurrent, activationFunc, addNodeBias, addNodeGain):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize = hiddenSize
        self.initialConnection = initialConnection
        self.maxWeight = maxWeight
        self.minWeight = minWeight
        self.initialWeightType = initialWeightType
        self.initialWeightMean = initialWeightMean
        self.initialWeightScale = initialWeightScale
        self.maxBias = maxBias
        self.minBias = minBias
        self.initialBiasType = initialBiasType
        self.initialBiasMean = initialBiasMean
        self.initialBiasScale = initialBiasScale
        self.maxStrategy = maxStrategy
        self.minStrategy = minStrategy
        self.strategy = initialStrategy
        self.isRecurrent = isRecurrent
        self.fitness = 0.0

        if initialStrategy < 0 or minStrategy < 0:
            raise ValueError("Strategy should be non-zero positive value")

        self.activationFunc = activationFunc
        self.addNodeBias = addNodeBias
        self.addNodeGain = addNodeGain

        if self.initialBiasType == 'gaussian':
            initialBiases = [random.normalvariate(self.initialBiasMean, self.initialBiasScale)
                             for i in range(self.outputSize + self.hiddenSize)]
        elif self.initialBiasType == 'cauchy':
            initialBiases = [self.initialBiasMean +
                             self.initialBiasScale * math.tan(math.pi * (random.random() - 0.5))
                             for i in range(self.outputSize + self.hiddenSize)]
        else:
            if self.initialBiasType != 'uniform':
                print("WARNING: undefined 'initialBiasType'. Using 'uniform' instead")
            initialBiases = [random.uniform(self.minBias, self.maxBias)
                             for i in range(self.outputSize + self.hiddenSize)]
        initialBiases = np.clip(initialBiases, self.minBias, self.maxBias)

        inputNodeList = [Node(id=i, type='input')
                         for i in range(self.inputSize)]
        outputNodeList = [Node(id=i + self.inputSize, type='output', bias=initialBiases[i])
                          for i in range(self.outputSize)]
        hiddenNodeList = [Node(id=i + self.inputSize + self.outputSize,
                          type='hidden', bias=initialBiases[i + self.outputSize])
                          for i in range(self.hiddenSize)]

        self.maxNodeID = self.inputSize + self.outputSize + self.hiddenSize - 1

        availableConnection = 0
        if self.hiddenSize == 0:
            availableConnection = self.inputSize * self.outputSize
        else:
            availableConnection = (self.inputSize * self.outputSize +
                                   self.inputSize * self.hiddenSize +
                                   self.outputSize * self.hiddenSize)

        connectionNumber = int(round(availableConnection * self.initialConnection))
        connectionIndex = random.sample(range(availableConnection), connectionNumber)

        connections = np.zeros(availableConnection)
        for i in connectionIndex:
            connections[i] = 1

        if self.initialWeightType == 'gaussian':
            initialWeights = [random.normalvariate(self.initialWeightMean, self.initialWeightScale)
                              for i in range(connectionNumber)]
        elif self.initialWeightType == 'cauchy':
            initialWeights = [self.initialWeightMean +
                              self.initialWeightScale * math.tan(math.pi * (random.random() - 0.5))
                              for i in range(connectionNumber)]
        else:
            if self.initialWeightType != 'uniform':
                print("WARNING: undefined 'initialWeightType'. Using 'uniform' instead")
            initialWeights = [random.uniform(self.minWeight, self.maxWeight)
                              for i in range(connectionNumber)]
        initialWeights = np.clip(initialWeights, self.minWeight, self.maxWeight)

        # Construct the initial topology.
        linkID = 0
        linkList = []
        disabledLinkList = []

        for i, (input, output) in enumerate(itertools.product(inputNodeList, outputNodeList)):
            if connections[i] == 1:
                linkList += [Link(id=linkID,
                                  fromNodeID=input.id,
                                  toNodeID=output.id,
                                  weight=initialWeights[linkID])]
                linkID += 1
            else:
                disabledLinkList += [(input.id, output.id)]

        # No output-to-output recurrent connections in the initial topology.
        if self.isRecurrent == True:
            for output in outputNodeList:
                disabledLinkList += [(output.id, output.id)]

        self.operonList = [Operon(id=0,
                                  nodeList=np.concatenate([inputNodeList, outputNodeList]),
                                  linkList=np.array(linkList))]
        self.operonList[0].setDisabledLinkList(disabledLinkList)
        maxOperonID = 0

        if self.hiddenSize != 0:
            # Initializing the network with hidden nodes is not recommended, as similarly discussed in the NEAT algorithm.
            # In addition, there are no mutations to reduce nodes or links.
            # ---
            # Note:
            # Each hidden node is assigned to a different operon.
            # Only feed-forward connections (input to hidden, hidden to output connections) are configured.
            # If the initial topology is not defined with full connections, some hidden nodes might have zero in/out-degree.
            # ---
            for i, hidden in enumerate(hiddenNodeList):
                maxOperonID = i + 1
                linkList = []
                disabledLinkList = []
                for j, input in enumerate(inputNodeList):
                    if connections[i * (self.inputSize + self.outputSize) + j +
                                   self.inputSize * self.outputSize] == 1:
                        linkList += [Link(id=linkID,
                                          fromNodeID=input.id,
                                          toNodeID=hidden.id,
                                          weight=initialWeights[linkID])]
                        linkID += 1
                    else:
                        disabledLinkList += [(input.id, hidden.id)]

                for j, output in enumerate(outputNodeList):
                    if connections[i * (self.inputSize + self.outputSize) + j +
                                   self.inputSize * self.outputSize + self.inputSize] == 1:
                        linkList += [Link(id=linkID,
                                          fromNodeID=hidden.id,
                                          toNodeID=output.id,
                                          weight=initialWeights[linkID])]
                        linkID += 1
                    else:
                        disabledLinkList += [(hidden.id, output.id)]
                    if self.isRecurrent == True:
                        disabledLinkList += [(output.id, hidden.id)]
                if self.isRecurrent == True:
                    disabledLinkList += [(hidden.id, hidden.id)]

                self.operonList += [Operon(id=maxOperonID,
                                           nodeList=np.array([hidden]),
                                           linkList=np.array(linkList))]
                self.operonList[maxOperonID].setDisabledLinkList(disabledLinkList)

        self.maxOperonID = maxOperonID
        self.maxLinkID = linkID - 1

    def getNodeAndLinkNum(self):
        nodeList, linkList = [], []
        for operon in self.operonList:
            nodeList = np.concatenate([nodeList, operon.nodeList])
            linkList = np.concatenate([linkList, operon.linkList])
        return len(nodeList), len(linkList)

    def calculateNetwork(self, inputsList):

        if len(inputsList) != self.inputSize:
            raise ValueError("The number of inputs doesn't match")

        nodeList, linkList = [], []
        for operon in self.operonList:
            nodeList = np.concatenate([nodeList, operon.nodeList])
            linkList = np.concatenate([linkList, operon.linkList])

        inputNodeList = sorted([n for n in nodeList if n.type == 'input'], key=lambda n: n.id)
        hiddenNodeList = sorted([n for n in nodeList if n.type == 'hidden'], key=lambda n: n.id)
        outputNodeList = sorted([n for n in nodeList if n.type == 'output'], key=lambda n: n.id)

        for node, input_val in zip(inputNodeList, inputsList):
            node.value = input_val

        weightMatrix = torch.zeros((self.maxNodeID + 1, self.maxNodeID + 1), device=device)
        for link in linkList:
            weightMatrix[link.toNodeID][link.fromNodeID] = link.weight

        nodeValueVec = torch.empty(self.maxNodeID + 1, dtype=torch.float, device=device)
        for node in nodeList:
            nodeValueVec[node.id] = node.value

        # Hidden layer processing.
        hiddenIDs = torch.tensor([i.id for i in hiddenNodeList], dtype=torch.long, device=device)
        hiddenBiases = torch.tensor([i.bias for i in hiddenNodeList], dtype=torch.float, device=device)
        hiddenNodeValueSum = weightMatrix[hiddenIDs] @ nodeValueVec

        if self.activationFunc == 'sigmoid':
            nodeValueVec[hiddenIDs] = torch.sigmoid(self.addNodeGain * (hiddenNodeValueSum - hiddenBiases))
        elif self.activationFunc == 'tanh':
            nodeValueVec[hiddenIDs] = torch.tanh(self.addNodeGain * (hiddenNodeValueSum - hiddenBiases))
        else:
            raise NameError(f"Activation function '{self.activationFunc}' is not defined")

        # Assign values to each hidden node class.
        for node in hiddenNodeList:
            node.value = nodeValueVec[node.id].item()

        # Output layer processing.
        outputIDs = torch.tensor([i.id for i in outputNodeList], dtype=torch.long, device=device)
        outputBiases = torch.tensor([i.bias for i in outputNodeList], dtype=torch.float, device=device)
        outputNodeValueSum = weightMatrix[outputIDs] @ nodeValueVec

        if self.activationFunc == 'sigmoid':
            nodeValueVec[outputIDs] = torch.sigmoid(self.addNodeGain * (outputNodeValueSum - outputBiases))
        elif self.activationFunc == 'tanh':
            nodeValueVec[outputIDs] = torch.tanh(self.addNodeGain * (outputNodeValueSum - outputBiases))
        else:
            raise NameError(f"Activation function '{self.activationFunc}' is not defined")

        # Assign values to each output node class.
        for node in outputNodeList:
            node.value = nodeValueVec[node.id].item()

        return nodeValueVec[outputIDs].tolist()


class ToolboxMBEANN:
    def __init__(self, p_addNode, p_addLink, p_weight, p_bias,
                 mutWeightType, mutWeightScale,
                 mutBiasType, mutBiasScale,
                 mutationProbCtl, addNodeWeight):
        self.p_addNode = p_addNode
        self.p_addLink = p_addLink
        self.p_weight = p_weight
        self.p_bias = p_bias
        self.mutWeightType = mutWeightType
        self.mutWeightScale = mutWeightScale
        self.mutBiasType = mutBiasType
        self.mutBiasScale = mutBiasScale
        self.mutationProbCtl = mutationProbCtl
        self.addNodeWeight = addNodeWeight

        if mutWeightType not in ['gaussian', 'cauchy', 'uniform', 'sa_one']:
            print("WARNING: undefined 'mutWeightType', using 'gaussian' instead")
            self.mutWeightType = 'gaussian'

        if mutBiasType not in ['gaussian', 'cauchy', 'uniform', 'sa_one']:
            print("WARNING: undefined 'mutBiasType', using 'gaussian' instead")
            self.mutBiasType = 'gaussian'

        # Warn if 'sa_one' is applied to the weights and biases independently.
        # To avoid updating the strategy parameter twice per generation.
        # Also, the learning parameter tau depends on the dimension.
        self.mutWeightTypeWarn = False
        self.mutateBiasValueWarn = False
        if self.mutWeightType == 'sa_one' and self.mutBiasType == 'sa_one':
            if self.p_bias > 0.0:
                self.mutWeightTypeWarn = True
            if self.p_weight > 0.0:
                self.mutateBiasValueWarn = True

    def mutateWeightValue(self, ind, c=1.0):

        if self.mutWeightType == 'sa_one' and self.p_weight > 0.0:
            # Warn if 'sa_one' is applied to the weights and biases independently.
            if self.mutWeightTypeWarn == True:
                print("WARNING: recommended to use 'mutateWeightAndBiasValue' instead of 'mutateWeightValue'")
                self.mutWeightTypeWarn = False
            N = random.normalvariate(0.0, 1.0)
            # Not sure if this works with tau decreasing depending on augmenting topologies.
            tau = c / math.sqrt(ind.maxLinkID + 1.0)
            ind.strategy *= math.exp(tau * N)
            ind.strategy = np.clip(ind.strategy, ind.minStrategy, ind.maxStrategy)

        for operon in ind.operonList:
            for link in operon.linkList:
                if random.random() < self.p_weight:
                    if self.mutWeightType == 'gaussian':
                        link.weight += self.mutWeightScale * random.normalvariate(0.0, 1.0)
                    elif self.mutWeightType == 'cauchy':
                        link.weight += self.mutWeightScale * math.tan(math.pi * (random.random() - 0.5))
                    elif self.mutWeightType == 'uniform':
                        link.weight = random.uniform(ind.minWeight, ind.maxWeight)
                    elif self.mutBiasType == 'sa_one':
                        link.weight += ind.strategy * random.normalvariate(0.0, 1.0)
                    link.weight = np.clip(link.weight, ind.minWeight, ind.maxWeight)

    def mutateBiasValue(self, ind, c=1.0):

        if self.mutBiasType == 'sa_one' and self.p_bias > 0.0:
            if self.mutateBiasValueWarn == True:
                print("WARNING: recommended to use 'mutateWeightAndBiasValue' instead of 'mutateBiasValue'")
                self.mutateBiasValueWarn = False
            N = random.normalvariate(0.0, 1.0)
            # Not sure if this works with tau decreasing depending on augmenting topologies.
            tau = c / math.sqrt(ind.maxNodeID + 1.0 - ind.inputSize)
            ind.strategy *= math.exp(tau * N)
            ind.strategy = np.clip(ind.strategy, ind.minStrategy, ind.maxStrategy)

        for operon in ind.operonList:
            for node in operon.nodeList:
                if node.type != 'input' and random.random() < self.p_bias:
                    if self.mutBiasType == 'gaussian':
                        node.bias += self.mutBiasScale * random.normalvariate(0.0, 1.0)
                    elif self.mutBiasType == 'cauchy':
                        node.bias += self.mutBiasScale * math.tan(math.pi * (random.random() - 0.5))
                    elif self.mutBiasType == 'uniform':
                        node.bias = random.uniform(ind.minBias, ind.maxBias)
                    elif self.mutBiasType == 'sa_one':
                        node.bias += ind.strategy * random.normalvariate(0.0, 1.0)
                    node.bias = np.clip(node.bias, ind.minBias, ind.maxBias)

    def mutateWeightAndBiasValue(self, ind, c=1.0):
        if self.mutWeightType == 'sa_one' and self.mutBiasType == 'sa_one':
            N = random.normalvariate(0.0, 1.0)
            # Not sure if this works with tau decreasing depending on augmenting topologies.
            tau = c / math.sqrt(ind.maxLinkID + ind.maxNodeID + 2.0 - ind.inputSize)
            ind.strategy *= math.exp(tau * N)
            ind.strategy = np.clip(ind.strategy, ind.minStrategy, ind.maxStrategy)
            for operon in ind.operonList:
                for link in operon.linkList:
                    if random.random() < self.p_weight:
                        link.weight += ind.strategy * random.normalvariate(0.0, 1.0)
                        link.weight = np.clip(link.weight, ind.minWeight, ind.maxWeight)

                for node in operon.nodeList:
                    if node.type != 'input' and random.random() < self.p_bias:
                        node.bias += ind.strategy * random.normalvariate(0.0, 1.0)
                        node.bias = np.clip(node.bias, ind.minBias, ind.maxBias)
        else:
            self.mutateWeightValue(ind)
            self.mutateBiasValue(ind)

    def mutateAddNode(self, ind, prob_lower_bound=None):
        # Normalize the mutation probability if mutationProbCtl = 'network'.
        if self.mutationProbCtl == 'network':
            mutation_prob = 1.0 - ((1.0 - self.p_addNode) ** (1.0 / (ind.maxOperonID + 1.0)))
            if prob_lower_bound != None:
                if prob_lower_bound > 1.0:
                    raise ValueError("'prob_lower_bound' should be smaller than 1.0")
                mutation_prob = np.clip(mutation_prob, prob_lower_bound, 1.0)
        else:
            mutation_prob = self.p_addNode
            if self.mutationProbCtl != 'operon':
                print("WARNING: undefined 'mutationProbCtl'. Using 'operon' instead")
            if prob_lower_bound != None:
                print("WARNING: cannot use 'prob_lower_bound' in the 'operon' setting")

        newOperonID = None

        for operon in ind.operonList:

            # If this operon is the newly generated operon.
            if operon.id == newOperonID:
                break

            if random.random() < mutation_prob:
                if len(operon.linkList) != 0:
                    randomIndex = random.randint(0, len(operon.linkList) - 1)
                else:
                    break
                oldLink = operon.linkList[randomIndex]

                operon.addLinkToDisabledLinkList(oldLink.fromNodeID, oldLink.toNodeID)
                operon.linkList = np.delete(operon.linkList, randomIndex)

                newNode = Node(id=ind.maxNodeID + 1,
                               type='hidden',
                               bias=ind.addNodeBias)
                newLinkA = Link(id=ind.maxLinkID + 1,
                                fromNodeID=ind.maxNodeID + 1,
                                toNodeID=oldLink.toNodeID,
                                weight=oldLink.weight)
                newLinkB = Link(id=ind.maxLinkID + 2,
                                fromNodeID=oldLink.fromNodeID,
                                toNodeID=ind.maxNodeID + 1,
                                weight=self.addNodeWeight)

                ind.maxNodeID += 1
                ind.maxLinkID += 2

                if operon.id == 0:
                    newOperonID = ind.maxOperonID + 1
                    ind.maxOperonID += 1
                    ind.operonList += [Operon(id=newOperonID,
                                              nodeList=np.array([newNode]),
                                              linkList=np.array([newLinkA, newLinkB]))]

                    disabledLinkList = []
                    if ind.isRecurrent == True:
                        for nodeOperon0 in operon.nodeList:
                            if nodeOperon0.type != 'input':
                                disabledLinkList += [(newNode.id, nodeOperon0.id)]
                            disabledLinkList += [(nodeOperon0.id, newNode.id)]
                        disabledLinkList += [(newNode.id, newNode.id)]
                    else:
                        for nodeOperon0 in operon.nodeList:
                            if nodeOperon0.type == 'input':
                                disabledLinkList += [(nodeOperon0.id, newNode.id)]
                            if nodeOperon0.type == 'output':
                                disabledLinkList += [(newNode.id, nodeOperon0.id)]

                    ind.operonList[newOperonID].setDisabledLinkList(disabledLinkList)
                    ind.operonList[newOperonID].deleteLinkFromDisabledLinkList(newLinkA.fromNodeID, newLinkA.toNodeID)
                    ind.operonList[newOperonID].deleteLinkFromDisabledLinkList(newLinkB.fromNodeID, newLinkB.toNodeID)

                else:
                    if ind.isRecurrent == True:
                        for nodeOperon0 in ind.operonList[0].nodeList:
                            if nodeOperon0.type != 'input':
                                operon.addLinkToDisabledLinkList(newNode.id, nodeOperon0.id)
                            operon.addLinkToDisabledLinkList(nodeOperon0.id, newNode.id)
                        for node in operon.nodeList:
                            operon.addLinkToDisabledLinkList(newNode.id, node.id)
                            operon.addLinkToDisabledLinkList(node.id, newNode.id)
                        operon.addLinkToDisabledLinkList(newNode.id, newNode.id)

                        operon.deleteLinkFromDisabledLinkList(newLinkA.fromNodeID, newLinkA.toNodeID)
                        operon.deleteLinkFromDisabledLinkList(newLinkB.fromNodeID, newLinkB.toNodeID)

                    else:
                        # TODO: Trying to delete a selected link that is already deleted.
                        for nodeOperon0 in ind.operonList[0].nodeList:
                            if nodeOperon0.type == 'input':
                                operon.addLinkToDisabledLinkList(nodeOperon0.id, newNode.id)
                            if nodeOperon0.type == 'output':
                                operon.addLinkToDisabledLinkList(newNode.id, nodeOperon0.id)

                    operon.nodeList = np.append(operon.nodeList, [newNode])
                    operon.linkList = np.append(operon.linkList, [newLinkA, newLinkB])

    def mutateAddLink(self, ind, prob_lower_bound=None):
        # Normalize the mutation probability if mutationProbCtl = 'network'.
        if self.mutationProbCtl == 'network':
            mutation_prob = 1.0 - ((1.0 - self.p_addLink) ** (1.0 / (ind.maxOperonID + 1.0)))
            if prob_lower_bound != None:
                if prob_lower_bound > 1.0:
                    raise ValueError("'prob_lower_bound' should be smaller than 1.0")
                mutation_prob = np.clip(mutation_prob, prob_lower_bound, 1.0)
        else:
            mutation_prob = self.p_addLink
            if self.mutationProbCtl != 'operon':
                print("WARNING: undefined 'mutationProbCtl'. Using 'operon' instead")
            if prob_lower_bound != None:
                print("WARNING: cannot use 'prob_lower_bound' in the 'operon' setting")

        for operon in ind.operonList:
            if len(operon.disabledLinkList) != 0:
                if random.random() < mutation_prob:
                    randomIndex = random.randint(0, len(operon.disabledLinkList) - 1)
                    newLinkFromNodeID = operon.disabledLinkList[randomIndex][0]
                    newLinkToNodeID = operon.disabledLinkList[randomIndex][1]
                    newLink = Link(id=ind.maxLinkID + 1,
                                   fromNodeID=newLinkFromNodeID,
                                   toNodeID=newLinkToNodeID,
                                   weight=0.0)
                    ind.maxLinkID += 1
                    operon.deleteLinkFromDisabledLinkList(newLinkFromNodeID, newLinkToNodeID)
                    operon.linkList = np.append(operon.linkList, [newLink])

    def selectionSettings(self, pop, popSize, isMaximizingFit, eliteSize=0):
        pop.sort(key=lambda ind: ind.fitness, reverse=isMaximizingFit)
        self.pop = pop
        self.popSize = popSize
        self.isMaximizingFit = isMaximizingFit
        self.eliteSize = eliteSize

    def preserveElite(self):
        return self.pop[0:self.eliteSize]

    def selectionRandom(self):
        newPop = random.choices(self.pop, k=self.popSize - self.eliteSize)
        newPop.sort(key=lambda ind: ind.fitness, reverse=self.isMaximizingFit)
        newPop = [copy.deepcopy(ind) for ind in newPop]
        return newPop

    def selectionTournament(self, tournamentSize, bestN=1):
        # Select "bestN" individuals from each tournament.
        if self.popSize < tournamentSize:
            raise ValueError("Population size should be smaller than tournament")
        if tournamentSize < bestN:
            raise ValueError("'bestN' should be smaller than tournament size")
        newPop = []
        while len(newPop) < self.popSize - self.eliteSize:
            tournament = random.sample(self.pop, tournamentSize)
            tournament.sort(key=lambda ind: ind.fitness, reverse=self.isMaximizingFit)
            newPop += tournament[0:bestN]
        newPop = [copy.deepcopy(ind) for ind in newPop]
        return newPop
