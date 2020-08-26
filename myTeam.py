# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).



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
import numpy as np

import pickle



# If you change these, you won't affect the server, so you can't cheat
KILL_POINTS = 0
SONAR_NOISE_RANGE = 13 # Must be odd
SONAR_NOISE_VALUES = [i - (SONAR_NOISE_RANGE - 1)/2 for i in range(SONAR_NOISE_RANGE)]
SIGHT_RANGE = 5 # Manhattan distance
MIN_FOOD = 2
TOTAL_FOOD = 60

DUMP_FOOD_ON_DEATH = True # if we have the gameplay element that dumps dots on death

SCARED_TIME = 40

def noisyDistance(pos1, pos2):
  return int(util.manhattanDistance(pos1, pos2) + random.choice(SONAR_NOISE_VALUES))

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

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
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


###############
# Ghost Agent #
###############
# copied from baselineTeam.py
class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    Ghost Agent that keeps our side PacMan free.
    Will try to eat opponent's PacMan by finding and going towards it.
    Will run away for the duration that it is scared/eatable.
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        # member variables
        # saves how many capsules are on our side left
        self.numCapsules = len(self.getCapsulesYouAreDefending(gameState))
        self.invaderDistance = []  # list of the most recent and exact coordinates of invaders
        self.middleOfBoard = tuple(map(lambda i, j: math.floor(
            (i+j)/2), gameState.data.layout.agentPositions[0][1], gameState.data.layout.agentPositions[1][1]))
        # print(self.middleOfBoard)
        # used if there is an enemy on the map whose exact position we do not know, but has eaten food recently.
        self.target = 0

    def getFeatures(self, gameState, action):
        """
        Gets features of the gameState after the arg action has been made
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman:
            features['onDefense'] = 0

        # Computes distance to invaders we can see (within 5 distance)
        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        numEnemies = len([a for a in enemies if a.isPacman])
        # holds the invaders that we can see
        invaders = [a for a in enemies if a.isPacman and a.getPosition()
                    != None]
        isStuck = self.isStuck(successor, action)
        if invaders:
            # store the most recent distance of the invader(s)
            self.invaderDistance = invaders
        elif self.invaderDistance and numEnemies and not isStuck:
            # use the most recent distance of the invader(s)
            invaders = self.invaderDistance
        elif numEnemies:
            eatenFood = self.guessPacManPosition(successor, action)
            if eatenFood:
                self.target = eatenFood
                self.invaderDistance = []
        else:
            invaders = []
            self.invaderDistance = []
            self.target = 0

        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(
                myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)
        elif numEnemies and self.target:
            features['invaderDistance'] = self.getMazeDistance(
                myPos, self.target)

        if numEnemies == 0:
            # if there is no enemies then camp at middle
            y = -1
            while gameState.hasWall(self.middleOfBoard[0], self.middleOfBoard[1]):
                # so that we don't throw an error that we can't go to that position because there is a wall
                # print(self.middleOfBoard)
                self.middleOfBoard = (
                    self.middleOfBoard[0], self.middleOfBoard[1]+y)
                if self.middleOfBoard[1] < 1:
                    y = 1
            features['invaderDistance'] = self.getMazeDistance(
                myPos, self.middleOfBoard)

        features['live'] = 1 if myPos != self.start else -1
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(
            self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
        # to prevent ghost from going back to the start position
        if self.red and myPos[0] <= self.start[0]+2:
            features['beginning'] = 1
        elif not self.red and myPos[0] >= self.start[0]-2:
            features['beginning'] = 1
        return features

    def evaluate(self, gameState, action):
        """
        Evaluate actions based on the game state and state of ghost.
        Depending on the state of the ghost, will evaluate the actions differently.
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features*weights

    def chooseAction(self, gameState):
        """
        Picks on an action that gives the highest evaluation.
        """
        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')
        values = [self.evaluate(gameState, a) for a in actions]

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        return random.choice(bestActions)

    def isStuck(self, gameState, action):
        """
        Checks to see if ghost gets stuck while trying to chase pacman
        """
        if len(self.observationHistory) > 6:
            # if we do not move an average of 2 distance over 6 moves, then assume we are stuck
            # get the 3rd to last move
            oldpos3 = self.observationHistory[-3].getAgentPosition(self.index)
            # get the 6th to last move
            oldpos6 = self.observationHistory[-6].getAgentPosition(self.index)
            currPos = gameState.getAgentPosition(self.index)
            dist = self.getMazeDistance(currPos, oldpos3)
            dist += self.getMazeDistance(oldpos3, oldpos6)
            return dist/2 <= 2  # calculate the average distance and see if we moved more than 2
        return False

    def guessPacManPosition(self, gameState, action):
        """
        Ghost Agent will try to guess where PamMan is based on the food that are missing from the board by comparing board states
        """
        foodPositions = self.getFoodYouAreDefending(gameState).asList()
        passedGameState = self.getPreviousObservation()
        if passedGameState:
            OldFood = self.getFoodYouAreDefending(passedGameState).asList()
            # find food position that was eaten
            foodEaten = [i for i in OldFood if i not in foodPositions]
            if foodEaten:
                return random.choice(foodEaten)
        return 0

    def getWeights(self, gameState, action):
        """
        Weights of each feature. In a normal ghost state, we will try to penalize the action that does not eat PacMan and increases distance to PacMan.
        We will also penalize the action that makes the ghost go back to its original starting position heavily.
        """
        if self.isScared(gameState):
            # see if it is more worth it to get eaten and return to current position vs stay alive
            if gameState.data.agentStates[self.index].scaredTimer > self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), self.start):
                return {'numInvaders': 0, 'onDefense': 100, 'invaderDistance': -20, 'stop': -30, 'reverse': -2, 'beginning': -400, 'live': -1000}
            else:
                return {'numInvaders': 0, 'onDefense': 100, 'invaderDistance': 0, 'stop': -30, 'reverse': -2, 'beginning': -400, 'live': 1000}
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -30, 'reverse': -2, 'beginning': -400, 'live': 0}

    def ourCapsuleEaten(self, gameState):
        """
        Helper function to see if a capsule on our side was eaten
        """
        # check if a capsule on our side was eaten
        boardCapsules = len(self.getCapsulesYouAreDefending(gameState))
        if (self.numCapsules > boardCapsules):
            # print("enemy pacman ate capsule")
            self.numCapsules -= 1
            return True
        return False

    def isScared(self, gameState):
        """
        Checks to see if our ghost is scared. Has a 40 move countdown to keep track of scared state
        """
        return gameState.data.agentStates[self.index].scaredTimer > 0


################
# Pacman Agent #
################
class OffensiveReflexAgent(ReflexCaptureAgent):
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
        self.deathCoord = None
        self.deathScore = 0
        self.pathChoices = []
        self.pathTaken = []
        self.middleOfBoard = tuple(map(lambda i, j: math.floor(
            (i+j)/2), gameState.data.layout.agentPositions[0][1], gameState.data.layout.agentPositions[1][1]))
        self.totalFood = len(self.getFood(gameState).asList())
       
        # for training data
        self.gameFeatures = [] # uncomment to retrain data
        # self.gameFeatures = readTrainingInputs()
        self.gameOutputs = []

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

    def getScoreIncrease(self, gameState):
        """
        returns how much we increased the score
        """
        if len(self.observationHistory) > 1:
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

    def checkDeath(self, gameState):
        """
        checks if pacman dies by seeing if we return back to start.
        """

        if len(self.observationHistory) > 0:
            currentPos = gameState.getAgentState(self.index).getPosition()
            # if we are at self.start:
            if currentPos == self.start:
                return 1
        return 0

    def getReward(self, gameState):
        """
        Gets the reward of the current gameState
        Score = rewards - punishment
        """
        score = 0
        # rewards
        score += 1.5 if self.ateFood(gameState) else -0.5  # add if we eat food
        # add the amount of score increase
        score += self.getScoreIncrease(gameState)*2
        score += 1 if gameState.getAgentState(self.index).isPacman else -1
        foodList = self.getFood(gameState).asList()

        if not self.ateFood(gameState):
            if foodList:
                minDistance = min([self.getMazeDistance(gameState.getAgentState(
                    self.index).getPosition(), food) for food in foodList])
                score = np.reciprocal(minDistance) # distance to closest food

        score += self.getEnemyDistance(gameState)
        # if the game is over big reward if we win, else penalty if we lose
        # punishment
        score -= 1 if self.checkDeath(gameState) else -0.5  # punish if we die

        return score

    def getFeatures(self, gameState, action):
        """
        Returns the feature of the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        # say that capsules are also food
        foodList += self.getCapsules(successor)
        features['successorScore'] = -len(foodList)  # self.getScore(successor)

        # Computes distance to invaders we can see (within 5 distance)
        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        invaders = [
            a for a in enemies if not a.isPacman and a.getPosition() != None]
        myPos = successor.getAgentState(self.index).getPosition()
        if len(invaders) > 0:
            dists = [self.getMazeDistance(
                myPos, a.getPosition()) for a in invaders]
            features['distanceToEnemy'] = min(dists) if min(dists) < 10 else 0
        # Compute distance to the nearest food
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            minDistance = min([self.getMazeDistance(myPos, food)
                               for food in foodList])
            features['distanceToFood'] = minDistance

        # N,E,S,W - available actions + STOP and REVERSE
        numWalls = 4 - len(successor.getLegalActions(self.index)) + 1

        features['numWalls'] = numWalls

        if self.numFoodCarrying >= 5 or len(foodList) <= 2:
            pos2 = successor.getAgentPosition(self.index)
            dist = self.getMazeDistance(self.start, pos2)
            features['distanceToHome'] = dist

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(
            self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        # Get the death coordinates
        self.getDeathCoordinates(gameState)

        # if there are death coordinates
        if self.deathCoord and self.eatOrRetreat(gameState):
            # set sail to those coordinates
            features['distanceToFood'] = self.getMazeDistance(myPos, self.deathCoord)
        return features

    def getWeights(self, gameState, action):
        """
        Gives Weights for PacMan gameState features.
        Penalize getting away from food and getting closer to enemy
        Penalize getting trapped between walls
        """
        if len(self.getFood(gameState).asList()) <= 2:
            # Return home and try to avoid ghosts
            return {'successorScore': 100, 'distanceToFood': -1, 'distanceToEnemy': 1.5, 'stop': -300, 'reverse': 0.1, 'numWalls': -0.5, 'distanceToHome': -1.5}
        if self.numFoodCarrying >= 5:
            # Do not care about getting more food.
            return {'successorScore': 100, 'distanceToFood': 0, 'distanceToEnemy': 1.5, 'stop': -300, 'reverse': 0.1, 'numWalls': -0.5, 'distanceToHome': -1}
        # TODO: figure out individual ghost scared weights
        if self.isScared(gameState, 0) and self.isScared(gameState, 1):
            return {'successorScore': 100, 'distanceToFood': -1, 'distanceToEnemy': -0.5, 'stop': -300, 'reverse': 0.1, 'numWalls': -0.5}
        return {'successorScore': 100, 'distanceToFood': -1, 'distanceToEnemy': 1.5, 'stop': -300, 'reverse': 0.1, 'numWalls': -0.5}

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
                # print("our pacman ate capsule")
                return True
        else:
            return False

    def distToFood(self, gameState):
        """
        Returns the distance to the closest food (capsules and dots) we can eat
        """
        foodList = self.getFood(gameState).asList()
        myPos = gameState.getAgentState(self.index).getPosition()
        # say that capsules are also food
        foodList += self.getCapsules(gameState)
        if self.ateFood(gameState):
            # print("ate food")
            return 0
        if len(foodList) > 0:  # This should always be True, but better safe than sorry
            return min([self.getMazeDistance(myPos, food) for food in foodList])
        return 0

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
                    if not currentGhostPosition or self.getMazeDistance(previousGhostPosition, currentGhostPosition) > 1:
                        # ghost is no longer scared after being eaten
                        self.scaredGhostTimers[ghostIndex] = 0
                        return True
        return False

    def distOurSide(self, gameState):
        """
        Calculates the distance to get back to our team's side as PacMan
        """
        if not gameState.getAgentState(self.index).isPacman:
            return 0
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        ourSide = (self.middleOfBoard[0], int(myPos[1])) # get the distance in terms of x, because the Y does not matter
        # to make sure that we are at our side
        #  in case there is a wall
        while gameState.hasWall(ourSide[0], ourSide[1]):
            ourSide = (ourSide[0] - 1, ourSide[1]
                       ) if self.red else (ourSide[0] + 1, ourSide[1])
        return self.getMazeDistance(myPos, ourSide)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)
        self.isCapsuleEaten(gameState)
        actionChosen = None
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        # decrement scaredTimers if needed.
        for i in range(len(self.scaredGhostTimers)):
            self.scaredGhostTimers[i] -= 1 if self.scaredGhostTimers[i] > 0 else 0

        foodLeft = len(self.getFood(gameState).asList())
        if self.ateFood(gameState):
            self.numFoodCarrying += 1
        if not gameState.getAgentState(self.index).isPacman:
            # if we are a ghost, then that means we are back on our side
            self.numFoodCarrying = 0

        if gameState.getAgentPosition(self.index):
            self.deathCoord = None
            self.start

        if len(self.pathTaken) > 0:
            # call choice function
            actionChosen = self.bestPath(gameState, bestActions)
        else:
            # else choose a random action
            actionChosen = bestActions[0]

        self.pathTaken.append(bestActions)
        successor = self.getSuccessor(gameState, actionChosen)
        # [x_0,x_1,x_2,...,x_7] = [minDistToEnemy, distFood, distOurSide,currentScore]
        stateFeatures = []
        stateFeatures.append(self.getEnemyDistance(successor))
        stateFeatures.append(self.distToFood(successor))
        stateFeatures.append(self.distOurSide(successor))
        stateFeatures.append(self.getScore(successor))
        if len(self.gameOutputs):
            # for training data outputs, reward and when game is finished add the game score
            stateOutput = [self.getReward(successor)]
        else:
            stateOutput = [self.getReward(successor)]
        self.gameFeatures.append(stateFeatures)
        self.gameOutputs.append(stateOutput)
        return actionChosen
    
    def bestPath(self, gameState, paths):
        """
        paths is the best actions array
        we need to choose the best one based on cost
        """
        bestChoices = [paths[0]]
        for p in paths:
            if p not in self.pathTaken:
                bestChoices.append(p)
        bestChoices.sort()
        return bestChoices[0]

    def isScared(self, gameState, ghostIndex):
        """
        Checks if a ghost, given the arg ghostIndex is in a scared state.
        A ghost is in a scared state if it's timer is greater than 0
        """
        return self.scaredGhostTimers[ghostIndex] > 0

    def ateFood(self, gameState):
        """
        Returns true if PacMan ate food in the last turn
        """
        previousObservation = self.getPreviousObservation()  # get the previous observation
        if previousObservation:
            # get previous turn number of food on enemy side
            previousFood = len(self.getFood(previousObservation).asList())
            foodLeft = len(self.getFood(gameState).asList())
            return previousFood > foodLeft
        return False

    def getDeathCoordinates(self, gameState):
        """
        This functions finds the location where pacman dies. These coordinates are needed
        for pacman to go back and attempt to get the food back.
        The default for death coordinates is null
        """
        previousState = self.getPreviousObservation()

        if previousState:
            if self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), previousState.getAgentState(self.index).getPosition()) > 1:
                self.deathCoord = self.start
                # self.deathScore = self.getScore
            # get the previous position
            previousGameState = self.getPreviousObservation()

            if previousGameState:
                # then check if we are currently at self.start
                currentPos = gameState.getAgentState(self.index).getPosition()
                # if we are at self.start:
                if currentPos == self.start:
                    self.deathCoord = previousGameState.getAgentPosition(
                        self.index)


    def eatOrRetreat(self, gameState):
        """
        This function checks if it is worth it to retreave what has been lost or to
        just play as normal
        """
        # failsafe default of 5
        foodLost = 5
        if self.observationHistory[-1]:
            # get the previos status of the game
            prevPrevFood = len(self.getFood(
                self.observationHistory[0]).asList())
            prevFood = len(self.getFood(self.observationHistory[-1]).asList())
            # if playing as red, calculate how much food we had available in the prev game on the blue side
            # else, do the same but on the red side
            if self.red:
                AgentStateScore = len(
                    self.getPreviousObservation().getBlueFood().asList())
            else:
                AgentStateScore = len(
                    self.getPreviousObservation().getRedFood().asList())
            # food lost is the food in current - last round
            foodLost = abs(AgentStateScore - prevFood)
        # if the amount of food >= 5 return true
        if foodLost >= 5:
            return True
        return False

    def final(self, gameState):
        """
        Puts the gathered training data from this match into input and output files.
        Will add the final score of the game before putting the output values into file
        """
        self.gameOutputs = [[i + self.getScore(gameState) for i in l] for l in self.gameOutputs]
        totalOutputs = []+self.gameOutputs
        # totalOutputs = readTrainingOutputs()+self.gameOutputs
        print(len(self.gameFeatures))
        print(len(totalOutputs))
        with open('./trainingInput.pickle', 'wb') as handle:
            pickle.dump(self.gameFeatures, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open('./trainingOutput.pickle', 'wb') as handle:
            pickle.dump(totalOutputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

