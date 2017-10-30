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

import math

class Node:

    Cp = 10

    def __init__(self, gameState, agentId, remainingSteps, action, parent):
        self.visits = 0
        self.reward = 0.0

        self.gameState = gameState
        self.agentId = agentId
        self.remainingSteps = remainingSteps

        self.children = []
        self.action = action
        self.parent = parent

    def isTerminalState(self):
        if self.gameState.isWin() or self.gameState.isLose():
            return True
        if self.remainingSteps <= 0:
            return True
        return False

    def isFullyExpanded(self):
        legalActions = len(self.gameState.getLegalActions(self.agentId))
        visitedChild = len(self.children)
        return legalActions == visitedChild

    def nextAgent(self):
        return (self.agentId + 1) % self.gameState.getNumAgents()

    def backPropagation(self, reward):
        currentNode = self
        while currentNode is not None:
            currentNode.reward += reward
            currentNode.visits += 1
            currentNode = currentNode.parent

    # if this state is not fully expanded, find a child to expand
    # and returns this child
    def expand(self):
        childStates = [child.gameState for child in self.children]
        legalActions = self.gameState.getLegalActions(self.agentId)
        unexploredActions = []

        for action in legalActions:
            successor = self.gameState.generateSuccessor(self.agentId, action)
            if successor not in childStates:
                unexploredActions.append(action)

        # chose randomly from unexplored actions
        action = random.choice(unexploredActions)
        successor = self.gameState.generateSuccessor(self.agentId, action)
        node = Node(successor, self.nextAgent(), self.remainingSteps, action, self)
        self.children.append(node)
        return node

    # if this node is fully expanded, choose according to UCB algorithm
    # returns the node selected
    def UCB(self):
        prob = util.Counter()
        for child in self.children:
            prob[child] = child.reward/child.visits + 2*self.Cp*math.sqrt(2*math.log(self.visits)/child.visits)
        prob.normalize()
        return util.chooseFromDistribution(prob)

    def bestAction(self):
        best = -999999
        action = None
        for child in self.children:
            prob = child.reward/child.visits
            if prob > best:
                action = child.action
                best = prob
        return action

    # iteratively find the start node to start simulation on
    # and returns this node
    # if the current Agent is Pacman, follow standard rules
    # if the current Agent is a Ghost, use Directional Ghost Probability
    def selectChild(self):
        selectedNode = None

        if self.isTerminalState():
            return self

        # current Agent is Pacman
        if self.agentId == 0:
            if self.isFullyExpanded():
                selectedNode = self.UCB().selectChild()
            else:
                selectedNode = self.expand()
        # current Agent is Ghost
        else:
            action = getDirectionalGhostAction(self.gameState, self.agentId)
            successor = self.gameState.generateSuccessor(self.agentId, action)
            for child in self.children:
                if child.gameState == successor:
                    selectedNode = child
                    return selectedNode

            node = Node(successor, self.nextAgent(), self.remainingSteps, action, self)
            self.children.append(node)
            selectedNode = node.selectChild()

        # print "selected" , selectedNode.gameState
        return selectedNode

    def simulate(self):
        node = self
        for i in xrange(self.remainingSteps):
            if node.isTerminalState():
                break
            if node.agentId == 0:
                legalActions = node.gameState.getLegalPacmanActions()
                action = random.choice(legalActions)
                successor = node.gameState.generateSuccessor(node.agentId, action)
                newNode = Node(successor, node.nextAgent(), node.remainingSteps-1, action, node)
                self.children.append(newNode)
                node = newNode
            else:
                action = getDirectionalGhostAction(node.gameState, node.agentId)
                successor = node.gameState.generateSuccessor(node.agentId, action)
                newNode = Node(successor, node.nextAgent(), node.remainingSteps-1, action, node)
                self.children.append(newNode)
                node = newNode
        # print "terminalNode", node.gameState
        node.backPropagation(node.gameState.getScore())



class ContestAgent(MultiAgentSearchAgent):
    """
      Your Monte Carlo Tree Search agent (question 6)
    """

    iterationSteps = 200

    def getAction(self, gameState):
        """
          Returns the Monte Carlo action using self.depth and self.evaluationFunction

          All ghosts should be modeled as chasing Pacman
        """
        "*** YOUR CODE HERE ***"
        print "Start..!!!", gameState
        remainingSteps = gameState.getNumAgents() * 15
        node = Node(gameState, 0, remainingSteps, None, None)
        for i in xrange(self.iterationSteps):
            rootNode = node.selectChild()
            # print "selectedNode", rootNode.gameState
            rootNode.simulate()

        return node.bestAction()


from game import Actions
def getDirectionalGhostAction( state, agentId ):
    "A ghost that prefers to rush Pacman, or flee when scared."
    index = agentId
    prob_attack = 0.8
    prob_scaredFlee = 0.8

    # Read variables from state
    ghostState = state.getGhostState( index )
    legalActions = state.getLegalActions( index )
    pos = state.getGhostPosition( index )
    isScared = ghostState.scaredTimer > 0

    speed = 1
    if isScared: speed = 0.5

    actionVectors = [Actions.directionToVector( a, speed ) for a in legalActions]
    newPositions = [( pos[0]+a[0], pos[1]+a[1] ) for a in actionVectors]
    pacmanPosition = state.getPacmanPosition()

    # Select best actions given the state
    distancesToPacman = [manhattanDistance( pos, pacmanPosition ) for pos in newPositions]
    if isScared:
        bestScore = max( distancesToPacman )
        bestProb = prob_scaredFlee
    else:
        bestScore = min( distancesToPacman )
        bestProb = prob_attack
    bestActions = [action for action, distance in zip( legalActions, distancesToPacman ) if distance == bestScore]

    # Construct distribution
    dist = util.Counter()
    for a in bestActions: dist[a] = bestProb / len(bestActions)
    for a in legalActions: dist[a] += ( 1-bestProb ) / len(legalActions)
    dist.normalize()

    if len(dist) == 0:
        return Directions.STOP
    else:
        return util.chooseFromDistribution(dist)



