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
import random, time, util
from game import Directions
import game
from util import nearestPoint
import itertools

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DummyAgent'):
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
  first = "JointParticleFilterAgent"
  second = "JointParticleFilterAgent"
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class CTFAgent(CaptureAgent):
  """
  A test class for agents
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
    start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

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
      return bestAction #chooses action that make you closest to your start state

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    debug = False
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if debug:
        print("pos " + str(pos))
        print("nearestPoint " + str(nearestPoint(pos)) + "\n")
        #The point of the if-else statement below is to make sure you are on a grid, because you could be between points
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


class OffensiveReflexAgent(CTFAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)  # self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0:  # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}


class DefensiveReflexAgent(CTFAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}


class JointParticleFilterAgent(CTFAgent):
  """
  JointParticleFilter tracks a joint distribution over tuples of all ghost
  positions.
  """

  def registerInitialState(self, gameState, numParticles=600):
    self.setNumParticles(numParticles)
    CTFAgent.registerInitialState(self, gameState)
    self.initialize(gameState)

  def setNumParticles(self, numParticles):
    self.numParticles = numParticles

  def initialize(self, gameState, legalPositions=None):
    "Stores information about the game, then initializes particles."
    self.numEnemies = gameState.getNumAgents() - 2
    self.enemies = []
    self.legalPositions = gameState.getWalls().asList(False)
    self.initializeParticles()

    # for fail
    self.initialGameState = gameState

  def initializeParticles(self):
    """
    Initialize particles to be consistent with a uniform prior.

    Each particle is a tuple of ghost positions. Use self.numParticles for
    the number of particles. You may find the `itertools` package helpful.
    Specifically, you will need to think about permutations of legal ghost
    positions, with the additional understanding that ghosts may occupy the
    same space. Look at the `itertools.product` function to get an
    implementation of the Cartesian product.

    Note: If you use itertools, keep in mind that permutations are not
    returned in a random order; you must shuffle the list of permutations in
    order to ensure even placement of particles across the board. Use
    self.legalPositions to obtain a list of positions a ghost may occupy.

    Note: the variable you store your particles in must be a list; a list is
    simply a collection of unweighted variables (positions in this case).
    Storing your particles as a Counter (where there could be an associated
    weight with each position) is incorrect and may produce errors.
    """
    positions = self.legalPositions
    enemy_positions = list(itertools.product(positions, positions))
    num_positions = len(enemy_positions)
    random.shuffle(enemy_positions)
    self.enemy_positions = enemy_positions

    atEach = self.numParticles / num_positions  # self.numParticles
    remainder = self.numParticles % num_positions
    # don't throw out a particle
    particles = []
    # populate particles
    for pos in enemy_positions:
      for num in range(atEach):
        particles.append(pos)
    # now populate the remainders
    for index in range(remainder):
      particles.append(enemy_positions[index])
    # save to self.particles
    random.shuffle(particles)
    self.particles = particles
    return particles

  def addEnemyAgent(self, agent):
    """
    Each ghost agent is registered separately and stored (in case they are
    different).
    """
    self.enemies.append(agent)

  def getJailPosition(self, i):  # need to pass in enemy index
    return self.initialGameState.getInitialAgentPosition(i)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    # You can profile your evaluation time by uncommenting these lines
    start = time.time()

    self.observeState(gameState)

    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]

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
      return bestAction #chooses action that make you closest to your start state

    print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    return random.choice(bestActions)

  def observeState(self, gameState):
    """
    Resamples the set of particles using the likelihood of the noisy
    observations.

    To loop over the ghosts, use:

      for i in range(self.numGhosts):
        ...

    A correct implementation will handle two special cases:
      1) When a ghost is captured by Pacman, all particles should be updated
         so that the ghost appears in its prison cell, position
         self.getJailPosition(i) where `i` is the index of the ghost.

         As before, you can check if a ghost has been captured by Pacman by
         checking if it has a noisyDistance of None.

      2) When all particles receive 0 weight, they should be recreated from
         the prior distribution by calling initializeParticles. After all
         particles are generated randomly, any ghosts that are eaten (have
         noisyDistance of None) must be changed to the jail Position. This
         will involve changing each particle if a ghost has been eaten.

    self.getParticleWithGhostInJail is a helper method to edit a specific
    particle. Since we store particles as tuples, they must be converted to
    a list, edited, and then converted back to a tuple. This is a common
    operation when placing a ghost in jail.
    """
    pacmanPosition = gameState.getAgentPosition(self.index)
    noisyDistances = gameState.getAgentDistances()  # gives noisy distances of ALL agents
    # emissionModels = [gameState.getDistanceProb(dist) for dist in noisyDistances]

    for enemy_num in range(2):#self.getOpponents(gameState):
      beliefDist = self.getBeliefDistribution()
      W = util.Counter()

      # JAIL? unhandled so far

      for p in self.particles:
        trueDistance = self.getMazeDistance(p[enemy_num], pacmanPosition)
        W[p] = (beliefDist[p] * gameState.getDistanceProb(trueDistance, noisyDistances[enemy_num]))

      # we resample after we get weights for each ghost
      if W.totalCount() == 0:
        self.particles = self.initializeParticles()
      else:
        values = []
        keys = []
        for key, value in W.items():
          keys.append(key)
          values.append(value)
        self.particles = util.nSample(values, keys, self.numParticles)

  def getBeliefDistribution(self):
    "*** YOUR CODE HERE ***"
    allPossible = util.Counter()
    for pos in self.particles:
      allPossible[pos] += 1
    allPossible.normalize()
    return allPossible

  def getEnemyPositions(self):
    """
    Uses getBeliefDistribution to predict where the two enemies are most likely to be
    :return: two tuples of enemy positions
    """
    return self.getBeliefDistribution().argMax()

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)

