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

# calculate reward by the transition of each state


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
        self.gamma = 0.90  # discount rate to prevent overfitting
        # is a counter of each state and action Q(s,a) = reward
        self.QValues = util.Counter()
        # self.QValues = readQValues()  # is a counter of each state and action Q(s,a) = reward
        self.scaredGhostTimers = [0,0]
        self.score = 0
        # for storing the action that we took and the past state
        # for some reson previousObservationHistory does not work
        self.previousGameStates = []
        self.previousActionTaken = []
        # self.features = util.Counter()
        self.weights = []
        self.weightInitialization()
        self.weights = readWeights()
        print(self.weights)

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
        self.QValues[(last_state, last_action)] = (reward + self.gamma + maxQ) * \
            self.alpha + (1-self.alpha)*self.QValues[(last_state, last_action)]
        

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
        Gets the best action given the current state. Find the action with the highest Q value using weights
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

    def isCapsuleEaten(self, gameState):
        """
        Checks if a capsule was eaten by our team pacman. If it was eaten, then this function also
        sets the enemy ghosts scared timer to 40
        """
        capsule = self.getCapsules(gameState)
        previousState =self.getPreviousObservation() # get the previous observation
        if previousState:
            previousCapsules = self.getCapsules(previousState)
        if len(capsule) != len(previousCapsules):
            self.scaredGhostTimers = [40,40] # both ghost's scared timers to 40 moves
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
        if self.isScared(gameState,ghostIndex):
            ghost = self.getOpponents(gameState)[ghostIndex] # get the ghost at the arg ghostIndex
            previousObservation = self.getPreviousObservation() # get the previous observation
            if previousObservation:
                previousGhostPosition = previousObservation.getAgentPosition(ghost)
            if previousGhostPosition:
                currentGhostPosition = gameState.getAgentPosition(ghost)
                # If we cannot find the ghost anymore, or if the ghost moved more than 1 position then the ghost
                # has been eaten.
                if not currentGhostPosition or self.getMazeDistance(previousGhostPosition,currentGhostPosition) > 1:
                    self.scaredGhostTimers[ghostIndex] = 0 # ghost is no longer scared after being eaten
                return True
        return False
        
    def getReward(self, gameState):
        """
        Gets the reward of the current gameState
        Score = rewards - punishment
        """
        score = 0
        # rewards
        # add if we eat the ghost
        score += 1.5 if self.ateFood(gameState) else 0
        score += self.getScoreIncrease(gameState)
        # punishment
        score -= 0.5 if self.checkDeath(gameState) else 0
        minDistance = min([self.getMazeDistance(gameState.getAgentState(
            self.index).getPosition(), food) for food in self.getFood(gameState).asList()])
        score += np.reciprocal(float(minDistance)) # distance to closest food
        # if the game is over big reward if we win, else penalty if we lose
        if gameState.isOver():
            score += self.getScore(gameState)*2
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
        """
        features of the state
        """
        successor = gameState.generateSuccessor(self.index, action)
        features = [1] # feature 0 is always 1
       
        features += [self.getEnemyDistance(successor)*0.1]  # distance to a visible enemy
        # features += [self.getMazeDistance(
        #     self.start, successor.getAgentState(self.index).getPosition())*0.3] # distance from start
        minDistance = min([self.getMazeDistance(successor.getAgentState(
            self.index).getPosition(), food) for food in self.getFood(successor).asList()])
        # choose action that decreases distance
        features += [np.reciprocal(float(minDistance))] # distance to closest food
        return features

    def calculateNewWeight(self, weight, delta, feature):
        # δ=r+γQ(s',a')-Q(s,a)
        # wi←wi+ηδFi(s,a).
        # print('weight', weight)
        weight = weight + self.alpha*delta*feature
        # print("new weight", weight)
        return weight

    def updateWeights(self, last_state, last_action, reward, max_q):
        # wi←wi+ηδFi(s,a).
        if (len(self.previousGameStates) > 0):
            delta = reward + self.gamma*max_q - self.getQValue(last_state, last_action)   # δ=r+γQ(s',a')-Q(s,a)
            for i in range(0, len(self.weights)):
                self.weights[i] = self.calculateNewWeight(
                    self.weights[i], delta, self.getFeatures(last_state, last_action)[i])

    def chooseAction(self, gameState):
        """
        Decides on the best action given the current state and looks at the Q table
        """
        actions = gameState.getLegalActions(self.index)
        if Directions.STOP in actions:
            # don't want to stop
            actions.remove(Directions.STOP)

        # reward of gameState to new gamestate
        # reward = self.Score(gameState)-self.score
        reward = self.getReward(gameState)
        # print('reward', reward)

        if len(self.previousGameStates) > 0:
            last_state = self.previousGameStates[-1]
            last_action = self.previousActionTaken[-1]
            max_q = self.getMaxQ(gameState)
            # self.updateQValue(last_state, last_action, reward, max_q)
            self.updateWeights(last_state, last_action, reward, max_q)

        # e-greedy
        if util.flipCoin(self.epsilon):
            action = random.choice(actions)
        else:
            action = self.bestAction(gameState)

        # update the score for the current state
        # self.score = self.Score(gameState)
        self.previousGameStates.append(gameState)
        self.previousActionTaken.append(action)

        # Check if we ate food --> then increase the food we are carrying
        successor = gameState.generateSuccessor(self.index, action)
        if self.ateFood(successor) == True:
            self.numFoodCarrying += 1

        # When the game is over, we need to update one last time and put into file
        if successor.isOver():
            self.final(gameState)

        # print('reward2', reward)
        return action

    def final(self, gameState):
        """
        Last Update of the Qvalues and pushes the Qvalue table back into file
        """
        print("game over. Weights:")
        print(self.weights)

        # put the Q values back into file
        # with open('./qValueFile.pickle', 'wb') as handle:
        #     pickle.dump(self.QValues, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # put weights into file
        with open('./LinearApproxFile.pickle', 'wb') as handle:
            pickle.dump(self.weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
