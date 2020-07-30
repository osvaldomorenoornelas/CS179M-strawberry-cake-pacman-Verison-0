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
# import gym

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



# init w = w_1,w_2,w_3,..w_n randomly in [0,1]
# for each episonde,
# 𝑠←initial state of episode
# 𝑎←action given by policy 𝜋 (recommend: 𝜖-greedy)
# Take action 𝑎, observe reward 𝑟 and next state 𝑠′
# 𝑤←𝑤+𝛼(𝑟+𝛾∗𝑚𝑎𝑥𝑎′𝑄(𝑠′,𝑎′)−𝑄(𝑠,𝑎))∇⃗ 𝑤𝑄(𝑠,𝑎)
# 𝑠←𝑠′
#
# class DeepQNetwork(nn.Module):
#     return

# An experience in SARSA of the form ⟨s,a,r,s',a'⟩ (the agent was in state s, did action a, and received reward r and ended up in state s', in which it decided to do action a')
# Q(s,a)≈θTϕ(s,a)
# Q(s,a) Sum(Features(state)*weight(action))
# each action gets a set of n weights to represent the q values for that action
# represent state,action with a function, instead of a table.
# gradient descent to find local min w = -0.5alpha*w*J(w)
# where J(w) is the loss
# 𝑄(𝑠,𝑎)=𝑤1𝑓1(𝑠,𝑎)+𝑤2𝑓2(𝑠,𝑎)+⋯,
# Q*(s,a) = Q*(s,a) + alpha(r+gamma*Q*(s',a) - Q*(s,a))
# for files
import pickle


def readQValues():
    """
    Will return the Counter of Q(s,a) from qValueFile
    """
    with open('./qValueFile.pickle', 'rb') as handle:
        qVals = pickle.load(handle)
    return qVals

def readWeights():
    """
    Return the list of weights from LinearApproxFile
    """
    with open('./LinearApproxFile.pickle', 'rb') as handle:
        weights = pickle.load(handle)
    return weights

def createTeam(firstIndex, secondIndex, isRed,
               first='QLearningAgent', second='DefensiveReflexAgent'):
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


class QLearningAgent(ReflexCaptureAgent):
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
        self.epsilon = 0.05  # exploration prob
        self.alpha = 0.1  # learning rate --> start with a large like 0.1 then exponentially smaller like 0.01, 0.001
        self.gamma = 0.25  # discount rate to prevent overfitting
        self.QValues = util.Counter()  # is a counter of each state and action Q(s,a) = reward
        self.QValues = readQValues()  # is a counter of each state and action Q(s,a) = reward
        
        self.score = 0
        # for storing the action that we took and the past state
        # for some reson previousObservationHistory does not work
        self.previousGameStates = []
        self.previousActionTaken = []
        # self.features = util.Counter()
        self.weights = []
# [0.9464804280010657, 0.41354557854314744, 0.44983954774355417, 0.3382974517826147, 0.8076823582245871, 0.8548595712604538, 0.814901663584643, 0.7614764948941833]
        self.weightInitialization()
        # self.weights = readWeights()
        print(self.weights)

    def getEnemyDistance(self, gameState):
        enemies = [gameState.getAgentState(i)
                   for i in self.getOpponents(gameState)]
        numEnemies = len([a for a in enemies if not a.isPacman])
        # holds the invaders that we can see
        invaders = [a for a in enemies if not a.isPacman and a.getPosition()
                    != None]
        if len(invaders) < 1:
            return 0
        dists = [self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), a.getPosition())
                 for a in invaders]
        return min(dists)

    
    def weightInitialization(self):
        """
        initializes 8 weights randomly from [0,1]. --> 8 Features = 8 weights
        Only call this ONCE --> for the first time running the training
        """
        self.weights = [random.random() for _ in range(8)]

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

    def getQValue(self, gameState, action):
        """
        Returns the Q value of the given state and action.
        If the state is not in the QValue, will return Q(s,a) = 0
        """
        # Qw(s,a) = w0+w1 F1(s,a) + ...+ wn Fn(s,a)
        Qval = 0
        features = self.getFeatures(gameState, action)
        for i in range(len(self.weights)):
            Qval += self.weights[i]*features[i]
        return Qval

    def updateQValue(self, last_state, last_action, reward, maxQ):
        """
        Updates the value for the gamestate and action in QTable 
        """
        # qVal = self.getQValue(gameState, action)
        # Q*(s,a) = (reward + discount + maxQ(s',a''))alpha + (1-alpha)*Q*(s,a)
        self.QValues[(last_state,last_action)] = (reward + self.gamma + maxQ)*self.alpha + (1-self.alpha)*self.QValues[(last_state,last_action)] 

        # self.QValues[(gameState,action)] = (reward + self.gamma + maxQ)*self.alpha + (1-self.alpha)*qVal
        # self.QValues[(gameState, action)] =(1 - self.alpha)*self.getQValue(gameState,action) + (reward + self.gamma + maxQ)*self.alpha

    def getMaxQ(self, gameState):
        """
        return the maximum Q of gameState
        """
        q_list = []
        for a in gameState.getLegalActions(self.index):
            q = self.getQValue(gameState, a)
            q_list.append(q)
        if len(q_list) == 0:
            return 0
        return max(q_list)

    def bestAction(self, gameState):
        """
        Gets the best action given the current state. Looks at the Q table and find the action with the highest value
        """
        actions = gameState.getLegalActions(self.index)
        if Directions.STOP in actions:
            # don't want to stop
            actions.remove(Directions.STOP)
        bestAction = actions[0]
        highestQVal = 0
        for action in actions:
            temp = self.getQValue(gameState, action)
            if temp > highestQVal:
                bestAction = action
                highestQVal = temp
        return bestAction

    def Score(self, gameState):
        """
        Gets the score of the current gameState
        Score = rewards - punishment
        """
        score = 0
        score += self.numFoodCarrying
        score += self.getScoreIncrease(gameState)
        score += self.checkDeath(gameState)*-3
        score += self.distToFood(gameState)*-3
        return score

    def ateFood(self, gameState):
        """
        Returns true if PacMan eats food in the turn
        """
        if len(self.previousGameStates) > 1:
            previousObservation = self.getPreviousObservation()  # get the previous observation
            # get previous turn number of food on enemy side
            previousFood = len(self.getFood(previousObservation).asList())
            foodLeft = len(self.getFood(gameState).asList())
            return previousFood != foodLeft
        return False

    def getFeatures(self, gameState, action):
            # self.features['ateFood']= 0.1 if self.ateFood(gameState) else 0
            # self.features['enemyDistance'] = self.getEnemyDistance(gameState)*0.1
            # self.features['carryingFood'] = self.numFoodCarrying*0.1
            # self.features['death'] = 0 if self.checkDeath(gameState)
            # self.features['increaseScore'] = getScoreIncrease(self.gameState)*0.1
            # self.features['distanceToSide'] = self.getMazeDistance(self.start, gameState.getAgentState(self.index).getPosition())*0.01
            successor = gameState.generateSuccessor(self.index, action)
            features = [1]
            features += [0.5 if self.ateFood(successor) else 0]
            features += [self.getEnemyDistance(successor)*0.3]
            features += [self.numFoodCarrying*0.1]
            # features += [0]
            features += [0 if self.checkDeath(successor) else 0.5]
            features += [self.getScoreIncrease(successor)]
            features += [self.getMazeDistance(
                self.start, successor.getAgentState(self.index).getPosition())*0.09]
            minDistance = min([self.getMazeDistance(successor.getAgentState(self.index).getPosition(), food) for food in self.getFood(successor).asList()])
        #    choose action that decreases distance
            features += [np.reciprocal(float(minDistance))*0.6]

            return features

    def calculateNewWeight(self, weight, delta, feature):
        # δ=r+γQ(s',a')-Q(s,a)
        # wi←wi+ηδFi(s,a).
        weight = weight + self.alpha*delta*feature
        return weight
    
    def updateWeights(self,last_state, last_action, reward, max_q):
        # wi←wi+ηδFi(s,a).
        # δ=r+γQ(s',a')-Q(s,a)
        if (len(self.previousGameStates)>0):
            delta = reward + self.gamma*max_q - self.getQValue(last_state, last_action)
            for i in range(0,len(self.weights)):
                self.weights[i] = self.calculateNewWeight(self.weights[i], delta, self.getFeatures(last_state,last_action)[i])
    
    def chooseAction(self, gameState):
        """
        Decides on the best action given the current state and looks at the Q table
        """
        actions = gameState.getLegalActions(self.index)
        if Directions.STOP in actions:
            # don't want to stop
            actions.remove(Directions.STOP)

        # reward of gameState
        reward = self.Score(gameState)-self.score

        if len(self.previousGameStates) > 0:
            last_state = self.previousGameStates[-1]
            last_action = self.previousActionTaken[-1]
            max_q = self.getMaxQ(gameState)
            self.updateQValue(last_state, last_action, reward, max_q)
            self.updateWeights(last_state, last_action, reward, max_q)

        # e-greedy
        if util.flipCoin(self.epsilon):
            action = random.choice(actions)
        else:
            action = self.bestAction(gameState)

        # update the score for the current state
        self.score = self.Score(gameState)
        self.previousGameStates.append(gameState)
        self.previousActionTaken.append(action)

        # Check if we ate food --> then increase the food we are carrying
        successor = gameState.generateSuccessor(self.index, action)
        if self.ateFood(successor) == True:
            self.numFoodCarrying += 1

        # When the game is over, we need to update one last time and put into file
        if successor.isOver():
            self.final(gameState)

        return action

    def final(self, gameState):
        """
        Last Update of the Qvalues and pushes the Qvalue table back into file
        """
        print("game over. Weights:")
        print(self.weights)
        reward = self.getScore(gameState)-self.score
        last_state = self.previousGameStates[-1]
        last_action = self.previousActionTaken[-1]
        self.updateQValue(last_state, last_action, reward, 0)

        # put the Q values back into file
        with open('./qValueFile.pickle', 'wb') as handle:
            pickle.dump(self.QValues, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # put weights into file
        with open('./LinearApproxFile.pickle', 'wb') as handle:
            pickle.dump(self.weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
