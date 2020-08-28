from captureAgents import CaptureAgent
import distanceCalculator
import random
import time
import util
import sys
import game
from game import Directions
from util import nearestPoint
import math
from myTeam import createTeam
from myTeam import ReflexCaptureAgent
from myTeam import DefensiveReflexAgent
from IPython.display import clear_output
import numpy as np
from DQNetwork import DQNetwork
import torch as T
import pickle
import os

def readDQInputs():
    """
    Will return the Counter of Q(s,a) from qValueFile
    """
    with open('./DQInput.pickle', 'rb') as handle:
        trainingInputs = pickle.load(handle)
    return trainingInputs


def readDQOutputs():
    """
    Return the list of weights from LinearApproxFile
    """
    with open('./DQOutput.pickle', 'rb') as handle:
        trainingOutputs = pickle.load(handle)
    return trainingOutputs


def createTeam(firstIndex, secondIndex, isRed,
               first='DQAgent', second='DefensiveReflexAgent'):
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


class DQAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        # initailize input data
        self.input_dims = 4  # 4 features
        self.out_dims = 1  # 1 output
        self.hidden_dimension = 2

        self.middleOfBoard = tuple(map(lambda i, j: math.floor(
            (i + j) / 2), gameState.data.layout.agentPositions[0][1], gameState.data.layout.agentPositions[1][1]))
        self.totalFood = len(self.getFood(gameState).asList())

        # declare Q-Value Network
        self.network = DQNetwork(self.input_dims, self.out_dims, self.hidden_dimension)

        # load the data
        print('get data')
        self.trainData, self.testData = self.network.loadData()
        print('Input size', self.sizeOfInput())
        print('Output size', self.sizeOfOutput())

        # train the model
        print('train data')
        self.network.Train(self.trainData, self.testData)
        print("Training successful")

        # for retraining the dq model
        self.gameFeatures = []
        # self.gameFeatures = readDQInputs()
        self.gameOutputs = []

    def sizeOfInput(self):
        """
        Returns the size of the training inputs file
        """
        return len(self.trainData)

    def sizeOfOutput(self):
        """
        Returns the size of the training outputs file
        """
        return len(self.testData)

    def getEnemyDistance(self, gameState):
        """
        Returns the distance to the nearest enemy. 
        If the enemy's exact location cannot be seen, then returns the closest noisy distance instead.
        """
        enemies = [gameState.getAgentState(i)
                   for i in self.getOpponents(gameState)]
        numEnemies = len([a for a in enemies if not a.isPacman])
        # holds the ghosts that we can see
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition()
                  != None]
        if len(ghosts) < 1:
            distances = [gameState.agentDistances[i]
                         for i in self.getOpponents(gameState)]
            # print(distances)
            # distances = [noisyDistance(pos, gameState.getAgentPosition(i)) for i in self.getOpponents(gameState)]
            return min(distances)
        dists = [self.getMazeDistance(gameState.getAgentState(
            self.index).getPosition(), a.getPosition()) for a in ghosts]
        return min(dists)

    def distToFood(self, gameState):
        """
        Returns the distance to the closest food (capsules and dots) we can eat
        """
        foodList = self.getFood(gameState).asList()
        myPos = gameState.getAgentState(self.index).getPosition()
        # say that capsules are also food
        foodList += self.getCapsules(gameState)
        if self.ateFood(gameState):
            return 0
        if len(foodList) > 0:  # This should always be True, but better safe than sorry
            return min([self.getMazeDistance(myPos, food) for food in foodList])
        return 0

    def ateFood(self, gameState):
        """
        Returns true if PacMan eats food in the previous turn
        """
        previousState = self.getPreviousObservation()

        if previousState:
            previousObservation = self.getPreviousObservation()  # get the previous observation
            # get previous turn number of food on enemy side
            previousFood = len(self.getFood(previousObservation).asList())
            previousFood += len(self.getCapsules(previousObservation))
            foodLeft = len(self.getFood(gameState).asList())
            foodLeft += len(self.getCapsules(gameState))
            return previousFood != foodLeft
        return False

    def distOurSide(self, gameState):
        """
        Returns the distance to get back to our team's side as a PacMan.
        Returns 0 if we are currently on our side
        """
        if not gameState.getAgentState(self.index).isPacman:
            return 0
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        ourSide = self.middleOfBoard
        # to make sure that we are at our side
        #  in case there is a wall
        while gameState.hasWall(ourSide[0], ourSide[1]):
            ourSide = (ourSide[0] - 1, ourSide[1]
                       ) if self.red else (ourSide[0] + 1, ourSide[1])
        return self.getMazeDistance(myPos, ourSide)

    def getFeatures(self, gameState, action):
        """
        return an array of features of the state
        """
        successor = gameState.generateSuccessor(self.index, action)
        stateFeatures = []
        stateFeatures.append(self.getEnemyDistance(successor))
        stateFeatures.append(self.distToFood(successor))
        stateFeatures.append(self.distOurSide(successor))
        stateFeatures.append(self.getScore(successor))
        return stateFeatures

    def getNetworkPrediction(self, features):
        """
        passes in a array of features into the neural network model and outputs a prediction 
        of the state's values
        """
        prediction = self.network.predict(features)
        return prediction

    def bestActionNN(self, gameState):
        """
        Returns the best action to take given the current game state
        """
        actions = gameState.getLegalActions(self.index)
        if Directions.STOP in actions:
            # don't want to stop
            actions.remove(Directions.STOP)
        bestAction = actions[0]
        highestQVal = 0
        for action in actions:
            # take in what the state action combination is
            features = self.getFeatures(gameState, action)
            outRes = self.getNetworkPrediction(features)
            temp = outRes[0][0]
            if temp > highestQVal:
                bestAction = action
                highestQVal = temp

        # for retraining model
        self.gameFeatures.append(self.getFeatures(gameState, bestAction))
        self.gameOutputs.append([highestQVal])
        return bestAction

    def chooseAction(self, gameState):
        """
        Decides on the best action given the current state and policy to return home after eating 20% of the food needed to win the game
        """
        action = self.bestActionNN(gameState)
        # if we have at least 20% of the food needed to win and we are tied or losing, bring it back home
        MIN_FOOD = 2
        foodToWin = (self.totalFood/2) - MIN_FOOD
        if gameState.data.agentStates[self.index].numCarrying >= math.ceil(foodToWin*.20) and self.getScore(gameState) <= 0:
            bestDist = 9999
            actions = gameState.getLegalActions(self.index)
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                dist = self.distOurSide(successor)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction
        return action

    def final(self, gameState):
        """
        For retraining the model, will add the final score to every output in the current match
        Then puts the inputs and outputs into respective files
        """
        self.gameOutputs = [[i + self.getScore(gameState) for i in l] for l in self.gameOutputs]
        totalOutputs = []+self.gameOutputs
        # totalOutputs = readDQOutputs()+self.gameOutputs

        print(len(self.gameFeatures))
        print(len(totalOutputs))
        with open('./DQInput.pickle', 'wb') as handle:
            pickle.dump(self.gameFeatures, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open('./DQOutput.pickle', 'wb') as handle:
            pickle.dump(totalOutputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
