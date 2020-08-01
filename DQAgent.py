import DQNetwork

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
import math


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
        self.scaredGhostTimers = [0, 0]  # timer for how long enemy ghost will be scared for
        self.numFoodCarrying = 0  # how much food pacman is carrying rn
        self.deathCoord = None
        self.deathScore = 0
        self.pathChoices = []
        self.pathTaken = []

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        foodList += self.getCapsules(successor)  # say that capsules are also food
        features['successorScore'] = -len(foodList)  # self.getScore(successor)

        # Computes distance to invaders we can see (within 5 distance)
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        myPos = successor.getAgentState(self.index).getPosition()
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['distanceToEnemy'] = min(dists) if min(dists) < 10 else 0
        # Compute distance to the nearest food
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        numWalls = 4 - len(successor.getLegalActions(self.index)) + 1  # N,E,S,W - available actions + STOP and REVERSE

        features['numWalls'] = numWalls

        if self.numFoodCarrying >= 5 or len(foodList) <= 2:
            pos2 = successor.getAgentPosition(self.index)
            dist = self.getMazeDistance(self.start, pos2)
            features['distanceToHome'] = dist

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        # Here we will go to the death coordinate

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
            return {'successorScore': 100, 'distanceToFood': -1, 'distanceToEnemy': 1.5, 'stop': -300, 'reverse': 0.1,
                    'numWalls': -0.5, 'distanceToHome': -1.5}
        if self.numFoodCarrying >= 5:
            # Do not care about getting more food.
            return {'successorScore': 100, 'distanceToFood': 0, 'distanceToEnemy': 1.5, 'stop': -300, 'reverse': 0.1,
                    'numWalls': -0.5, 'distanceToHome': -1}
        # TODO: figure out individual ghost scared weights
        if self.isScared(gameState, 0) and self.isScared(gameState, 1):
            return {'successorScore': 100, 'distanceToFood': -1, 'distanceToEnemy': -0.5, 'stop': -300, 'reverse': 0.1,
                    'numWalls': -0.5}
        return {'successorScore': 100, 'distanceToFood': -1, 'distanceToEnemy': 1.5, 'stop': -300, 'reverse': 0.1,
                'numWalls': -0.5}

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
                self.scaredGhostTimers = [40, 40]  # both ghost's scared timers to 40 moves
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
            ghost = self.getOpponents(gameState)[ghostIndex]  # get the ghost at the arg ghostIndex
            previousObservation = self.getPreviousObservation()  # get the previous observation
            if previousObservation:
                previousGhostPosition = previousObservation.getAgentPosition(ghost)
                if previousGhostPosition:
                    currentGhostPosition = gameState.getAgentPosition(ghost)
                    # If we cannot find the ghost anymore, or if the ghost moved more than 1 position then the ghost
                    # has been eaten.
                    if not currentGhostPosition or self.getMazeDistance(previousGhostPosition,
                                                                        currentGhostPosition) > 1:
                        self.scaredGhostTimers[ghostIndex] = 0  # ghost is no longer scared after being eaten
                        return True
        return False

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

        # for a in range(len(bestActions)):
        # print(str(a))
        # decrement scaredTimers if needed.
        for i in range(len(self.scaredGhostTimers)):
            self.scaredGhostTimers[i] -= 1 if self.scaredGhostTimers[i] > 0 else 0

        foodLeft = len(self.getFood(gameState).asList())
        if self.ateFood(gameState):
            self.numFoodCarrying += 1
        if not gameState.getAgentState(self.index).isPacman:
            # if we are a ghost, then that means we are back on our side
            self.numFoodCarrying = 0

        # I haven't tested to see if PacMan performs well with his 'return home' weights when there are only 2 food left.
        # (don't delete)
        # if foodLeft <= 2:
        #   bestDist = 9999
        #   for action in actions:
        #     successor = self.getSuccessor(gameState, action)
        #     pos2 = successor.getAgentPosition(self.index)
        #     dist = self.getMazeDistance(self.start,pos2)
        #     if dist < bestDist:
        #       bestAction = action
        #       bestDist = dist
        #   return bestAction
        if gameState.getAgentPosition(self.index):
            self.deathCoord = None
            self.start

        if len(self.pathTaken) > 0:
            # call choice function
            # print("Here len(self.pathTaken) > 0")
            actionChosen = self.bestPath(gameState, bestActions)
        else:
            # else choose a random action
            # print("Here choose first Action")
            actionChosen = bestActions[0]
            # actionChosen = random.choice(bestActions)

        # self.pathTaken.append(actionChosen)
        self.pathTaken.append(bestActions)

        # print(actionChosen)
        return actionChosen

    """
    paths is the best actions array
    we need to choose the best one based on cost
    """

    def bestPath(self, gameState, paths):

        # print("Here  bestPath(self,gameState, paths)")
        # print(len(paths))

        bestChoices = [paths[0]]

        for p in paths:
            if p not in self.pathTaken:
                # print("pruned")
                bestChoices.append(p)

        bestChoices.sort()
        # print("returned Val: ", bestChoices[0])
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
            previousFood = len(
                self.getFood(previousObservation).asList())  # get previous turn number of food on enemy side
            foodLeft = len(self.getFood(gameState).asList())
            return previousFood != foodLeft
        return False

    def getDeathCoordinates(self, gameState):
        """
        This functions finds the location where pacman dies. These coordinates are needed
        for pacman to go back and attempt to get the food back.
        The default for death coordinates is null
        """
        previousState = self.getPreviousObservation()

        if previousState:
            if self.getMazeDistance(gameState.getAgentState(self.index).getPosition(),
                                    previousState.getAgentState(self.index).getPosition()) > 1:
                self.deathCoord = self.start
                # self.deathScore = self.getScore
            # get the previous position
            previousGameState = self.getPreviousObservation()

            if previousGameState:
                # then check if we are currently at self.start
                currentPos = gameState.getAgentState(self.index).getPosition()
                # if we are at self.start:
                if currentPos == self.start:
                    self.deathCoord = previousGameState.getAgentPosition(self.index)

    """
    This fuction checks if it is worth it to retreave what has been lost or to 
    just play as normal
    """

    def eatOrRetreat(self, gameState):
        # get the amout of food that was lost

        # failsafe default of 5
        foodLost = 5

        if self.observationHistory[-1]:
            # get the previos status of the game
            prevPrevFood = len(self.getFood(self.observationHistory[0]).asList())
            prevFood = len(self.getFood(self.observationHistory[-1]).asList())

            # if playing as red, calculate how much food we had available in the prev game on the blue side
            # else, do the same but on the red side
            if self.red:
                AgentStateScore = len(self.getPreviousObservation().getBlueFood().asList())
            else:
                AgentStateScore = len(self.getPreviousObservation().getRedFood().asList())

            # food lost is the food in current - last round
            foodLost = abs(AgentStateScore - prevFood)

        # if the amount of food >= 5 return true
        if foodLost >= 5:
            return True

        return False