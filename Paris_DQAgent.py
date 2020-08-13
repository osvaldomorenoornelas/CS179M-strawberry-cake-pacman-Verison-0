import DQNetwork

from captureAgents import CaptureAgent
import distanceCalculator
import random
import time
import util
import sys
from game import Directions
import game
from util import nearestPoint
import math
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.nn import Module
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
import numpy as np
from torch.optim import SGD
from sklearn.metrics import accuracy_score

from myTeam import createTeam
from myTeam import ReflexCaptureAgent
from myTeam import DefensiveReflexAgent

import pickle


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='DQPacmanAgent', second='DefensiveReflexAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

# copied from baselineTeam.py
class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}

####################
# PAC-MAN DQ AGENT #
####################


class DQPacmanAgent(ReflexCaptureAgent):
    """
      A reflex agent that seeks food. This is an agent
      we give you to get an idea of what an offensive agent might look like,
      but it is by no means the best or only way to build an offensive agent.
      """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        # timer for how long enemy ghost will be scared for
        self.scaredGhostTimers = [0, 0]
        self.numFoodCarrying = 0  # how much food pacman is carrying rn

        # DQN variables
        self.gamma = 0.8 # discount factor
        self.epsilon = 0.05 # probability exploration
        self.actions = 5  # number of actions --> N,S,E,W,STOP
        self.inputs = 3 # input size
        self.outputs = 1 # output size

        # data batch input size
        # self.batch = batch_size

        # hidden dimension
        self.hidden = 2

        self.model = self.Model()

        self.weights = []
        self.weightInitialization()

        self.Data = []

        self.previousGameStates = []
        self.previousActionTaken = []

        # self.loadData()
        # print(len(self.Data))
        print(self.weights)
        self.Features = list()

        # self.evaluate(self.Data, self.model)
        # self.evaluate([([1,2,3],4),([4,5,6],7)], self.model)
        # prediction = self.predict([1,2,3], self.model)
        # print(prediction)
        
    def weightInitialization(self):
        """
        initializes 3 weights randomly from [0,1]. --> 3 Features = 3 weights
        Only call this ONCE --> for the first time running the training
        """
        self.weights = [random.random() for _ in range(3)]

    def distToFood(self, gameState):
        """
        Returns the distance to the closest food (capsules and dots) we can eat
        """
        foodList = self.getFood(gameState).asList()
        myPos = gameState.getAgentState(self.index).getPosition()
        # say that capsules are also food
        foodList += self.getCapsules(gameState)
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            return min([self.getMazeDistance(myPos, food) for food in foodList])
        return 0

    def getEnemyDistance(self, gameState):
        if gameState.getAgentState(self.index).isPacman:
            enemies = [gameState.getAgentState(i)
                       for i in self.getOpponents(gameState)]
            numEnemies = len([a for a in enemies if not a.isPacman])
            # holds the ghosts that we can see
            ghosts = [a for a in enemies if not a.isPacman and a.getPosition()
                      != None]
            if len(ghosts) < 1:
                return 0
            dists = [self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), a.getPosition())
                     for a in ghosts]
            return min(dists)
        return 0

    def isCapsuleEaten(self, gameState):
        """
        Checks if a capsule was eaten by our team pacman. If it was eaten, then this function also
        sets the enemy ghosts scared timer to 40
        """
        capsule = self.getCapsules(gameState)
        previousState = self.getPreviousObservation()  # get the previous observation
        if previousState:
            previousCapsules = self.getCapsules(previousState)
            if len(capsule) != len(previousCapsules):
                # both ghost's scared timers to 40 moves
                self.scaredGhostTimers = [40, 40]
                print("our pacman ate capsule")
                return True
        else:
            return False

    def isGhostEaten(self, gameState, ghostIndex):
        """
        Checks if the ghost in arg ghostIndex was eaten yet during the scared state.
        There is no need to check if a ghost was eaten if it already has a value of 0 in
        its scaredGhostTimer index.
        """
        if self.isScared(gameState, ghostIndex):
            # get the ghost at the arg ghostIndex
            ghost = self.getOpponents(gameState)[ghostIndex]
            previousObservation = self.getPreviousObservation()  # get the previous observation
            if previousObservation:
                previousGhostPosition = previousObservation.getAgentPosition(
                    ghost)
                if previousGhostPosition:
                    currentGhostPosition = gameState.getAgentPosition(ghost)
                    # If we cannot find the ghost anymore, or if the ghost moved more than 1 position then the ghost
                    # has been eaten.
                    if not currentGhostPosition or self.getMazeDistance(previousGhostPosition,
                                                                        currentGhostPosition) > 1:
                        # ghost is no longer scared after being eaten
                        self.scaredGhostTimers[ghostIndex] = 0
                        return True
        return False

    def isScared(self, gameState, ghostIndex):
        """
        Checks if a ghost, given the arg ghostIndex is in a scared state.
        A ghost is in a scared state if it's timer is greater than 0
        """
        return gameState.data.agentStates[ghostIndex].scaredTimer

    def checkDeath(self, gameState):
        """
        checks if pacman dies by seeing if we return back to start.
        """

        if len(self.previousGameStates) > 0:
            currentPos = gameState.getAgentState(self.index).getPosition()
            # if we are at self.start:
            if currentPos == self.start:
                return 1
        return 0

    def ateFood(self, gameState):
        """
        Returns true if PacMan ate food in the last turn
        """
        previousObservation = self.getPreviousObservation()  # get the previous observation
        if previousObservation:
            previousFood = len(
                self.getFood(previousObservation).asList())  # get previous turn number of food on enemy side
            foodLeft = len(self.getFood(gameState).asList())
            return previousFood != foodLeft
        return False

    def getScoreIncrease(self, gameState):
        """
        returns how much we increased the score
        """
        if len(self.previousGameStates) > 0:
            previousState = self.getPreviousObservation()
            score = self.getScore(gameState)
            prevScore = self.getScore(previousState)
            if prevScore != score:
                if self.red:
                    # if we get points as red then score increases
                    increase = score-prevScore
                    return increase if increase > 0 else 0
                else:
                    # if we get points as blue then score decrease
                    increase = score-prevScore
                    return increase if increase > 0 else 0
            return 0
        return 0

    def getReward(self, gameState):
            """
            Gets the reward of the current gameState
            Score = rewards - punishment
            """
            score = 0
            # rewards
            # add if we eat the ghost
            score += 1.25 if self.ateFood(gameState) else 0
            score += self.getScoreIncrease(gameState)
            # punishment
            score -= self.numFoodCarrying if self.checkDeath(gameState) else 0
            foodList = self.getFood(gameState).asList()
            if len(foodList) > 0:
                minDistance = min([self.getMazeDistance(gameState.getAgentState(
                    self.index).getPosition(), food) for food in foodList])
                score += np.reciprocal(float(minDistance)) # distance to closest food
            # if the game is over big reward if we win, else penalty if we lose
            if gameState.isOver():
                score += self.getScore(gameState)*2
            return score

    def getFeatures(self, gameState, action):
        """
        Two features to look at: distance to an enemy and distance to food
        """
        successor = gameState.generateSuccessor(self.index, action)
 
        features = [1]  # first feature is always 1
        features += [self.getEnemyDistance(successor)*0.1]
        features += [self.distToFood(successor)*0.1]
        return features

    def getQValue(self, gameState, action):
        """
        Returns the QValue of the given state and action.
        If the state is not in the QValue, will return Q(s,a) = 0.
        Q Value yielded from being at state s and selecting action a, is the immediate reward received, r(s,a), plus the highest Q Value possible from state s’ (which is the state we ended up in after taking action a from state s). 
        """
        # Qw(s,a) = r(s,a) + gamma*max_aQ(s',a)
        successor = self.getSuccessor(gameState, action)
        reward = self.getReward(successor)
        maxQNextState = 0
        # get max Q value of next state after taking that action
        legalActions = successor.getLegalActions(self.index)
        for a in legalActions:
            nextSuccessor = self.getSuccessor(successor, a)
            QNextState = self.getReward(nextSuccessor)
            if QNextState > maxQNextState:
                maxQNextState = QNextState
        Qval = reward + self.gamma*maxQNextState
        return Qval

        # Qw(s,a) = w0+w1 F1(s,a) + ...+ wn Fn(s,a)
        # Qval = 0
        # features = self.getFeatures(gameState, action)
        # for i in range(len(self.weights)):
        #     Qval += self.weights[i]*features[i]
        return Qval

    def loadData(self):
        """
        Load the data into x and y
        (data sets as shown by professor shelton on office hours)
        """
        # create tensors from the data file
        # for the moment, we will create temporaries from random
        with open('./DQNWeights.pickle', 'rb') as handle:
            self.weights = pickle.load(handle)
        with open('./DQNData.pickle', 'rb') as handle:
            self.Data = pickle.load(handle)
        # x = T.randn(self.actions, self.inputs)
        # y = T.randn(self.actions, self.outputs)
        # return x, y

    def Model(self):
        """
        alternate way to create the model without defining an MLP
        """
        model = T.nn.Sequential(T.nn.Linear(self.inputs, self.hidden),
                                T.nn.ReLU(),
                                T.nn.Linear(self.hidden, self.outputs),)
        return model

    def LossFunction(self):
        """
        Loss determines how good the weights are
        MSELoss measures the mean squared error
        """
        loss_fn = T.nn.MSELoss()
        return loss_fn

    def TrainModel(self, trainer, model):
        """
        To train the model we need to optamize compute, get loss, and
        update
        """

        print(model.parameters())

        # optamization algorithm
        criterion = self.LossFunction()

        # give SDG learning rate and momentum
        # SDG: Implements stochastic gradient descent
        optimizer = SGD(model.parameters(), lr=0.01)

        # train 50 iterations
        for epoch in range(50):
            # give minibatches umertaions
            for i, (inputs, targets) in enumerate(trainer):
                print("trainer")

                # zero_grad clears old gradients from the last step
                optimizer.zero_grad()

                # get the model output
                model_output = model(inputs)

                # obtain loss from model and targets
                loss = criterion(model_output, targets)

                # pass data back
                loss.backward()
                print(loss)
                # updates the parameters for gradietns (loss.backward)
                optimizer.step()


    def evaluate(self, testData, model):
        """
        This function evaluates the model and returns accuracy
        """
        # containers for the predictions and actual data
        predictions = list()
        actuals = list()

        accuracy = 0

        for i, (input, target) in enumerate(testData):
            input = Tensor(input)
            
            # print('inp',input)
            # input = T.stack(input).to(device)
            # print('tar',target)
            # evaluate data set model
            modelEval = model(input)
            # print(modelEval)
            # detach(): constructs a new view on a tensor
            # assigns it as a numpy array
            modelEval = modelEval.detach().numpy()
            # get numpy array of targets
            # actualData = target.numpy()
            actualData = np.float32(target)

            # tensor with the same data and number of elements as
            # input, but with the specified shape
            # actualData = actualData.reshape(len(actualData), 1)
            actualData = actualData.reshape(1, 1)
            # print(actualData)

            # # round class values
            # modelEval = modelEval.round()

            predictions.append(modelEval)
            actuals.append(actualData)

            # Stack arrays in sequence vertically (row wise)
            predictions = np.vstack(predictions)
            print(predictions)
            actuals = np.vstack(actuals)
            print(actuals)

            # get the accuracy (will improve as iterations occur)
            # accuracy = accuracy_score(actuals, predictions, normalize=True)
            accuracy = accuracy_score(actualData, modelEval, normalize=True)
            print(accuracy)

        return accuracy

    def predict(self, rowData, model):
        """
        Predicts for a row of data. May have to be more specific because
        of how pacman is structured
        """
        # convert row to data
        # rowData = Tensor([rowData])
        rowData = Tensor(rowData)

        # create a prediction
        modelData = model(rowData)

        # get the numpy Data
        modelData = modelData.detach().numpy()

        return modelData

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        # evaluate each gameState
        bestAction = actions[0]
        maxQ = 0
        for action in actions:
            # get features
            self.getFeatures(gameState, action)
            qVal = self.getQValue(gameState,action)
            if  qVal > maxQ:
                maxQ = qVal
                bestAction = action
            # evaluate and decide on the best one
        # Check if we ate food --> then increase the food we are carrying
        successor = self.getSuccessor(gameState, bestAction)
        if self.ateFood(successor) == True:
            self.numFoodCarrying += 1
        self.previousGameStates.append(gameState)
        self.previousActionTaken.append(action)
        # keep the features and max Q value
        # Data is list of tuples
        self.Data.append((self.getFeatures(gameState,bestAction),maxQ))
        self.Features.append(Tensor(self.getFeatures(gameState,bestAction)))

        # predict = self.predict(self.Features[-1], self.model)
        # print(predict)
       self.TrainModel()

        return bestAction        

    def final(self, gameState):
        """
        Last Update of the Qvalues and pushes the Qvalue table back into file
        """
        # self.TrainModel(self.Data,self.model)
        # print('eval',self.evaluate(self.Data,self.model))

        print("game over. Weights:")
        print(self.weights)
        

        with open('./DQNWeights.pickle', 'wb') as handle:
            pickle.dump(self.weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open('./DQNData.pickle', 'wb') as handle:
        #     pickle.dump(self.Data, handle, protocol=pickle.HIGHEST_PROTOCOL)