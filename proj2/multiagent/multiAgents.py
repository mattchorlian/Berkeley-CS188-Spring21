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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
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

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # distances to food  
        # if distance to ghost increased = good
        # if less food = good
        food_utility = 0
        ghost_utility = 0 
        ghost_list = []
        food_list = newFood.asList()
        food_dists = []
        ghost_dists = []


        for ghost in newGhostStates:
            ghost_list.append((ghost.getPosition()[0], ghost.getPosition()[1]))

        for food in food_list:
            food_dists.append(util.manhattanDistance(food, newPos))

        for ghost in ghost_list:
            ghost_dists.append(util.manhattanDistance(ghost, newPos))

       
        

        if newPos in ghost_list and (newScaredTimes[0] > 0):
            return 1
        
        if newPos in ghost_list and (newScaredTimes[0] <= 0):
            return -1
        
        if newPos in currentGameState.getFood().asList():
            return 1

        return 1/min(food_dists) - 1/(min(ghost_dists))

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
        self.best_move = None

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

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        best_score, best_move = self.minimax_decisions(gameState, depth = self.depth, agent = 0)
        return best_move


    def minimax_decisions(self, gameState, depth, agent):
        if agent == 0:
            move = self.maximizing_agent(gameState, agent, depth)
        else:
            move = self.minimizing_agent(gameState, agent, depth)
        
        return move

    def minimizing_agent(self, gameState, agent, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP
        actions = []
        actions = gameState.getLegalActions(agent)
        
        min_score = 100000000000
        minimized_action = Directions.STOP
        for action in actions:
            childGameState = gameState.getNextState(agent, action)
            if (agent == gameState.getNumAgents() - 1):
                utility = self.minimax_decisions(childGameState, depth - 1, 0)[0]
            else:
                utility = self.minimax_decisions(childGameState, depth, agent + 1)[0]
            if (utility < min_score):
                minimized_action = action 
                min_score  = utility
        return min_score, minimized_action


    def maximizing_agent(self, gameState, agent, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP
        actions = []
        actions = gameState.getLegalActions(agent)
        maximized_action = Directions.STOP
        max_score = -10000000000
        for action in actions:
            childGameState = gameState.getNextState(agent, action)
            if agent == gameState.getNumAgents() - 1:
                utility = self.minimax_decisions(childGameState, depth - 1, 0)[0]
            else:
                utility = self.minimax_decisions(childGameState, depth, agent + 1)[0]
            if (utility > max_score):
                maximized_action = action
                max_score = utility

        return max_score, maximized_action
        
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        best_score, best_move = self.alpha_beta_decisions(gameState, depth = self.depth, agent = 0, 
                                                alpha = -100000000, beta = 100000000)
        return best_move


    def alpha_beta_decisions(self, gameState, depth, agent, alpha, beta):
        if agent == 0:
            move = self.maximizing_agent(gameState, agent, depth, alpha, beta)
        else:
            move = self.minimizing_agent(gameState, agent, depth, alpha, beta)
        
        return move

    def minimizing_agent(self, gameState, agent, depth, alpha, beta):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP
        actions = []
        actions = gameState.getLegalActions(agent)
        
        min_score = 100000000000
        minimized_action = Directions.STOP
        for action in actions:
            childGameState = gameState.getNextState(agent, action)
            if (agent == gameState.getNumAgents() - 1):
                utility = self.alpha_beta_decisions(childGameState, depth - 1, 0, alpha, beta)[0]
            else:
                utility = self.alpha_beta_decisions(childGameState, depth, agent + 1, alpha, beta)[0]

            #pruning check   
            if (utility < alpha):
                return utility, action
                
            if (utility < min_score):
                minimized_action = action 
                min_score  = utility
            
            #update beta 
            beta = min(beta, utility)

        return min_score, minimized_action


    def maximizing_agent(self, gameState, agent, depth, alpha, beta):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP
        actions = []
        actions = gameState.getLegalActions(agent)
        max_score = -10000000
        maximized_action = Directions.STOP
        for action in actions:
            childGameState = gameState.getNextState(agent, action)
            if agent == gameState.getNumAgents() - 1:
                utility = self.alpha_beta_decisions(childGameState, depth - 1, 0, alpha, beta)[0]
            else:
                utility = self.alpha_beta_decisions(childGameState, depth, agent + 1, alpha, beta)[0]

            #pruning check
            if (utility > beta):
                return utility, action

            if (utility > max_score):
                maximized_action = action
                max_score = utility

            #update alpha
            alpha = max(alpha, utility)

        return max_score, maximized_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getAction(self, gameState):
        best_score, best_move = self.expectimax_decisions(gameState, depth = self.depth, agent = 0)
        return best_move


    def expectimax_decisions(self, gameState, depth, agent):
        if agent == 0:
            move = self.maximizing_agent(gameState, agent, depth)
        else:
            move = self.chance_agent(gameState, agent, depth)
        
        return move

    def chance_agent(self, gameState, agent, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP
        actions = []
        utility = 0
        actions = gameState.getLegalActions(agent)
        
        if agent == gameState.getNumAgents() - 1:
            for action in actions:
                childGameState = gameState.getNextState(agent, action)
                utility += (1/len(actions)) * (self.expectimax_decisions(childGameState, depth - 1, 0))[0]
                move = random.choice(actions)

        else:
            for action in actions:
                childGameState = gameState.getNextState(agent,action)
                utility += (1/len(actions)) * (self.expectimax_decisions(childGameState, depth, agent + 1))[0]
                move = random.choice(actions)

        return utility, move





    def minimizing_agent(self, gameState, agent, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP
        actions = []
        actions = gameState.getLegalActions(agent)
        
        min_score = 100000000000
        minimized_action = Directions.STOP
        for action in actions:
            childGameState = gameState.getNextState(agent, action)
            if (agent == gameState.getNumAgents() - 1):
                utility = self.expectimax_decisions(childGameState, depth - 1, 0)[0]
            else:
                utility = self.expectimax_decisions(childGameState, depth, agent + 1)[0]
            if (utility < min_score):
                minimized_action = action 
                min_score  = utility
        return min_score, minimized_action


    def maximizing_agent(self, gameState, agent, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP
        actions = []
        actions = gameState.getLegalActions(agent)
        maximized_action = Directions.STOP
        max_score = -10000000000
        for action in actions:
            childGameState = gameState.getNextState(agent, action)
            if agent == gameState.getNumAgents() - 1:
                utility = self.expectimax_decisions(childGameState, depth - 1, 0)[0]
            else:
                utility = self.expectimax_decisions(childGameState, depth, agent + 1)[0]
            if (utility > max_score):
                maximized_action = action
                max_score = utility

        return max_score, maximized_action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    food_dists = []
    ghost_list = []
    ghost_dists = []
    food_dist_total = 0
    ghost_dist_total = 0
    num_agents = currentGameState.getNumAgents()
    pac_state = currentGameState.getPacmanPosition()
    ghost_states = currentGameState.getGhostStates()
    legal_actions = currentGameState.getLegalActions()
    food_list = currentGameState.getFood().asList()
    newScaredTimes = [ghostState.scaredTimer for ghostState in ghost_states]


    for ghost in ghost_states:
        ghost_list.append((ghost.getPosition()[0], ghost.getPosition()[1]))

    for food in food_list:
        food_dists.append(util.manhattanDistance(food, pac_state))
        food_dist_total += util.manhattanDistance(food, pac_state)

    for ghost in ghost_list:
        ghost_dists.append(util.manhattanDistance(ghost, pac_state))
        ghost_dist_total += util.manhattanDistance(ghost, pac_state)

    if len(ghost_dists) > 0:
        closest_ghost = min(ghost_dists)

    if len(food_list) > 0:
        closest_food = min(food_dists)
        furthest_food = max(food_dists)

    total_food = len(food_list)
    if newScaredTimes[0] <= 0 and pac_state in ghost_list:
        return -100000000
    if newScaredTimes[0] > 0 and pac_state in ghost_list:
        return 1000000
    if total_food == 0:
        return 1000000000

    ghost_dist_avg = 1/len(ghost_list) * ghost_dist_total
    food_dist_avg = 1/len(food_list) * food_dist_total

    return currentGameState.getScore() + ghost_dist_total - food_dist_total + 400/(total_food + .01)



# Abbreviation
better = betterEvaluationFunction
