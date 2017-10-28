# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        score = successorGameState.getScore()

        # Closest Food Distance
        dist2Food = 999999
        for food in newFood.asList():
            tmpDist = manhattanDistance(newPos, food)
            if tmpDist < dist2Food:
                dist2Food = tmpDist
        if newFood.count > 0:
            score += 1.0 / dist2Food

        # Ghost Position
        ghostPos = newGhostStates[0].getPosition()
        dist2Ghost = manhattanDistance(newPos, ghostPos)
        if dist2Ghost > 0:
            score -= 2.0 / dist2Ghost

        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        return self.getValue(gameState, (0, 1))[1]
        #util.raiseNotDefined()

    def isLeafNode(self, gameState, depth):
        return (depth > self.depth) or gameState.isLose() or gameState.isWin()

    def getNext(self, gameState, agentId, depth):
        agentId += 1
        if agentId == gameState.getNumAgents():
            agentId = 0
            depth += 1
        return agentId, depth

    def getValue(self, gameState, (agentId, depth)):
        if self.isLeafNode(gameState, depth):
            #print (agentId, depth)
            return self.evaluationFunction(gameState), None
        if agentId == 0:
            return self.maxValue(gameState, agentId, depth)
        elif agentId < gameState.getNumAgents():
            return self.minValue(gameState, agentId, depth)
        else:
            raise Exception("Invalid AgentId")

    def maxValue(self, gameState, agentId, depth):
        v = -999999
        act = None
        legalActions = gameState.getLegalActions(agentId)
        for action in legalActions:
            successor = gameState.generateSuccessor(agentId, action)
            tmpValue = self.getValue(successor, self.getNext(gameState, agentId, depth))[0]
            if tmpValue > v:
                v = tmpValue
                act = action
        return v, act
        #util.raiseNotDefined()

    def minValue(self, gameState, agentId, depth):
        v = 999999
        act = None
        legalActions = gameState.getLegalActions(agentId)
        for action in legalActions:
            successor = gameState.generateSuccessor(agentId, action)
            tmpValue = self.getValue(successor, self.getNext(gameState, agentId, depth))[0]
            if tmpValue < v:
                v = tmpValue
                act = action
        return v, act
        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.getValue(gameState, (0, 1), -999999, 999999)[1]
        util.raiseNotDefined()

    def isLeafNode(self, gameState, depth):
        return (depth > self.depth) or gameState.isLose() or gameState.isWin()

    def getNext(self, gameState, agentId, depth):
        agentId += 1
        if agentId == gameState.getNumAgents():
            agentId = 0
            depth += 1
        return agentId, depth

    def getValue(self, gameState, (agentId, depth), alpha, beta):
        if self.isLeafNode(gameState, depth):
            #print (agentId, depth)
            return self.evaluationFunction(gameState), None
        if agentId == 0:
            return self.maxValue(gameState, agentId, depth, alpha, beta)
        elif agentId < gameState.getNumAgents():
            return self.minValue(gameState, agentId, depth, alpha, beta)
        else:
            raise Exception("Invalid AgentId")

    def maxValue(self, gameState, agentId, depth, alpha, beta):
        v = -999999
        act = None
        legalActions = gameState.getLegalActions(agentId)
        for action in legalActions:
            successor = gameState.generateSuccessor(agentId, action)
            tmpValue = self.getValue(successor, self.getNext(gameState, agentId, depth), alpha, beta)[0]
            if tmpValue > v:
                v = tmpValue
                act = action
            if v > beta:
                return v, act
            alpha = max(alpha, v)
        return v, act
        #util.raiseNotDefined()

    def minValue(self, gameState, agentId, depth, alpha, beta):
        v = 999999
        act = None
        legalActions = gameState.getLegalActions(agentId)
        for action in legalActions:
            successor = gameState.generateSuccessor(agentId, action)
            tmpValue = self.getValue(successor, self.getNext(gameState, agentId, depth), alpha, beta)[0]
            if tmpValue < v:
                v = tmpValue
                act = action
            if v < alpha:
                return v, act
            beta = min(beta, v)
        return v, act
        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.getValue(gameState, (0, 1))[1]
        util.raiseNotDefined()

    def isLeafNode(self, gameState, depth):
        return (depth > self.depth) or gameState.isLose() or gameState.isWin()

    def getNext(self, gameState, agentId, depth):
        agentId += 1
        if agentId == gameState.getNumAgents():
            agentId = 0
            depth += 1
        return agentId, depth

    def getValue(self, gameState, (agentId, depth)):
        if self.isLeafNode(gameState, depth):
            #print (agentId, depth)
            return self.evaluationFunction(gameState), None
        if agentId == 0:
            return self.maxValue(gameState, agentId, depth)
        elif agentId < gameState.getNumAgents():
            return self.expValue(gameState, agentId, depth)
        else:
            raise Exception("Invalid AgentId")

    def maxValue(self, gameState, agentId, depth):
        v = -999999
        act = None
        legalActions = gameState.getLegalActions(agentId)
        for action in legalActions:
            successor = gameState.generateSuccessor(agentId, action)
            tmpValue = self.getValue(successor, self.getNext(gameState, agentId, depth))[0]
            if tmpValue > v:
                v = tmpValue
                act = action
        return v, act
        #util.raiseNotDefined()

    def expValue(self, gameState, agentId, depth):
        v = 0
        legalActions = gameState.getLegalActions(agentId)
        for action in legalActions:
            successor = gameState.generateSuccessor(agentId, action)
            tmpValue = self.getValue(successor, self.getNext(gameState, agentId, depth))[0]
            v += tmpValue * (1.0/len(legalActions))
        return v, None
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    score = currentGameState.getScore()

    # Closest Food Distance
    dist2Food = 999999
    for food in newFood.asList():
        tmpDist = manhattanDistance(newPos, food)
        if tmpDist < dist2Food:
            dist2Food = tmpDist
    if newFood.count > 0:
        score += 1.0 / dist2Food

    # Ghost Position
    for ghost in newGhostStates:
        dist = manhattanDistance(newPos, ghost.getPosition())
        if dist == 0:
            continue
        if ghost.scaredTimer > 0:
            score += 10.0 / dist
        else:
            score -= 2.0 / dist

    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction


class ContestAgent(MultiAgentSearchAgent):
    """
      Your Monte Carlo Tree Search agent (question 6)
    """

    def getAction(self, gameState):
        """
          Returns the Monte Carlo action using self.depth and self.evaluationFunction

          All ghosts should be modeled as chasing Pacman
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
