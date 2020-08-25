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
from DQNetwork import MLP
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

def readTrainingInputs():
    """
    Will return the Counter of Q(s,a) from qValueFile
    """
    with open('./trainingInput.pickle', 'rb') as handle:
        trainingInputs = pickle.load(handle)
    return trainingInputs


def readTrainingOutputs():
    """
    Return the list of weights from LinearApproxFile
    """
    with open('./trainingOutput.pickle', 'rb') as handle:
        trainingOutputs = pickle.load(handle)
    return trainingOutputs


def readWeights():
    """
    Return the list of weights from LinearApproxFile
    """
    with open('./LinearApproxFile.pickle', 'rb') as handle:
        weights = pickle.load(handle)
    return weights


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
        self.numFoodCarrying = 0

        # Q Value functions
        self.epsilon = 0.005  # exploration prob
        self.alpha = 0.01
        self.gamma = 0.8  # discount rate to prevent overfitting

        # initailize input data
        self.n_actions = 5
        self.input_dims = 4  # 4 features
        self.out_dims = 1  # 1 output
        self.batch_size = 0
        self.hidden_dimension = 2

        self.middleOfBoard = tuple(map(lambda i, j: math.floor(
            (i + j) / 2), gameState.data.layout.agentPositions[0][1], gameState.data.layout.agentPositions[1][1]))
        self.totalFood = len(self.getFood(gameState).asList())

        # declare Q-Value Network

        self.network = DQNetwork(self.gamma, self.epsilon, self.n_actions, self.input_dims,
                                 self.out_dims, self.batch_size, self.hidden_dimension)

        # load the data
        print('get data')
        self.trainData, self.testData = self.network.loadData()
        print('Input size', self.sizeOfInput())
        print('Output size', self.sizeOfOutput())

        # train the model
        print('train data')
        self.network.Train(self.trainData, self.testData)
        print("Training succesful")

        # for retraining the dq model
        # self.gameFeatures = []
        self.gameFeatures = readDQInputs()
        self.gameOutputs = []

    def sizeOfInput(self):
        return len(self.trainData)

    def sizeOfOutput(self):
        return len(self.testData)

    def getEnemyDistance(self, gameState):
        enemies = [gameState.getAgentState(i)
                    for i in self.getOpponents(gameState)]
        numEnemies = len([a for a in enemies if not a.isPacman])
        # holds the ghosts that we can see
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if len(ghosts) < 1:
            distances = [ gameState.agentDistances[i] for i in self.getOpponents(gameState)]
            # print(distances)
            # distances = [noisyDistance(pos, gameState.getAgentPosition(i)) for i in self.getOpponents(gameState)]
            return min(distances)
        dists = [self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), a.getPosition()) for a in ghosts]
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

    def getNetworkPrediction(self, features):
        # make a prediction
        # pass in a set of features to be ran thru the model
        prediction = self.network.predict(features)
        # return prediction.round()
        return prediction

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
        score -= 1 if self.checkDeath(gameState) else 0

        foodList = self.getFood(gameState).asList()
        if len(foodList):
            minDistance = min([self.getMazeDistance(gameState.getAgentState(
                self.index).getPosition(), food) for food in foodList])
            # distance to closest food
            score += np.reciprocal(float(minDistance))
        # if the game is over big reward if we win, else penalty if we lose
        if gameState.isOver():
            score += self.getScore(gameState) * 2
        return score

    def ateFood(self, gameState):
        """
        Returns true if PacMan eats food in the turn
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

    def distToInvader(self, gameState):
        """
        Returns distance to an invader
        """
        myPos = gameState.getAgentState(self.index).getPosition()
        enemies = [gameState.getAgentState(i)
                   for i in self.getOpponents(gameState)]
        numEnemies = len([a for a in enemies if a.isPacman])
        # holds the invaders that we can see
        invaders = [a for a in enemies if a.isPacman and a.getPosition()
                    != None]
        if len(invaders):
            dists = [self.getMazeDistance(
                myPos, a.getPosition()) for a in invaders]
            return min(dists)
        return 0

    def getNumWalls(self, gameState):
        """
        Returns the number of walls in our way.
        This would be the 5 - number of legal actions at our current state
        Because stop is always a legal action
        """
        return 5 - len(gameState.getLegalActions(self.index))

    def getFeatures(self, gameState, action):
        """
        features of the state
        """
        successor = gameState.generateSuccessor(self.index, action)

        stateFeatures = []
        stateFeatures.append(self.getEnemyDistance(successor))
        stateFeatures.append(self.distToFood(successor))
        stateFeatures.append(self.distOurSide(successor))
        # stateFeatures.append(self.distToInvader(successor))
        stateFeatures.append(self.getScore(successor))
        # stateFeatures.append(self.getNumWalls(successor))
        # stateFeatures.append(int(successor.getAgentState(self.index).isPacman))
        # stateFeatures.append(
            # successor.data.agentStates[self.index].numCarrying)

        return stateFeatures

    def bestActionNN(self, gameState):
        actions = gameState.getLegalActions(self.index)
        if Directions.STOP in actions:
            # don't want to stop
            actions.remove(Directions.STOP)
        bestAction = actions[0]
        highestQVal = 0
        for action in actions:
            #print("action: ", action)
            # take in what the state action combination is
            features = self.getFeatures(gameState, action)
            outRes = self.getNetworkPrediction(features)
            temp = outRes[0][0]
            # print("Temp: ", temp)
            # print("Highest Val: ", highestQVal)
            if temp > highestQVal:
                bestAction = action
                highestQVal = temp


        # retrain model
        self.gameFeatures.append(self.getFeatures(gameState,bestAction))
        self.gameOutputs.append([highestQVal])
        return bestAction

    def chooseAction(self, gameState):
        """
        Decides on the best action given the current state and looks at the Q table
        """
        action = self.bestActionNN(gameState)
        # if we have at least 20% of the food needed to win and we are tied or losing, bring it back home
        MIN_FOOD = 2
        foodToWin = (self.totalFood/2) - MIN_FOOD
        if gameState.data.agentStates[self.index].numCarrying >= math.ceil(foodToWin*.20)  and self.getScore(gameState) <= 0:
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
            self.gameOutputs = [[i + self.getScore(gameState) for i in l] for l in self.gameOutputs]
            # print(self.gameOutputs)
            # totalOutputs = []+self.gameOutputs
            totalOutputs = readDQOutputs()+self.gameOutputs
            
            print(len(self.gameFeatures))
            print(len(totalOutputs))
            with open('./DQInput.pickle', 'wb') as handle:
                pickle.dump(self.gameFeatures, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
            with open('./DQOutput.pickle', 'wb') as handle:
                pickle.dump(totalOutputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    