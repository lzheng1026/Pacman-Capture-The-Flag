# inference.py
# ------------
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


import itertools
import util
import random
import game

from myTeam import CTFAgent


class JointParticleFilterAgent(CTFAgent):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """

    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
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
        pacmanPosition = self.index
        noisyDistances = gameState.getAgentDistances()  # gives noisy distances of ALL agents
        # emissionModels = [gameState.getDistanceProb(dist) for dist in noisyDistances]

        for enemy_num in self.getOpponents(gameState):
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

    # def getParticleWithGhostInJail(self, particle, ghostIndex):
    #     """
    #     Takes a particle (as a tuple of ghost positions) and returns a particle
    #     with the ghostIndex'th ghost in jail.
    #     """
    #     particle = list(particle)
    #     particle[ghostIndex] = self.getJailPosition(ghostIndex)
    #     return tuple(particle)
    # def elapseTime(self, gameState):  # need to figure out how to get the new probability distribution
    #     """
    #     Samples each particle's next state based on its current state and the
    #     gameState.
    #
    #     To loop over the ghosts, use:
    #
    #       for i in range(self.numGhosts):
    #         ...
    #
    #     Then, assuming that `i` refers to the index of the ghost, to obtain the
    #     distributions over new positions for that single ghost, given the list
    #     (prevGhostPositions) of previous positions of ALL of the ghosts, use
    #     this line of code:
    #
    #       newPosDist = getPositionDistributionForGhost(
    #          setGhostPositions(gameState, prevGhostPositions), i, self.ghostAgents[i]
    #       )
    #
    #     Note that you may need to replace `prevGhostPositions` with the correct
    #     name of the variable that you have used to refer to the list of the
    #     previous positions of all of the ghosts, and you may need to replace `i`
    #     with the variable you have used to refer to the index of the ghost for
    #     which you are computing the new position distribution.
    #
    #     As an implementation detail (with which you need not concern yourself),
    #     the line of code above for obtaining newPosDist makes use of two helper
    #     functions defined below in this file:
    #
    #       1) setGhostPositions(gameState, ghostPositions)
    #           This method alters the gameState by placing the ghosts in the
    #           supplied positions.
    #
    #       2) getPositionDistributionForGhost(gameState, ghostIndex, agent)
    #           This method uses the supplied ghost agent to determine what
    #           positions a ghost (ghostIndex) controlled by a particular agent
    #           (ghostAgent) will move to in the supplied gameState.  All ghosts
    #           must first be placed in the gameState using setGhostPositions
    #           above.
    #
    #           The ghost agent you are meant to supply is
    #           self.ghostAgents[ghostIndex-1], but in this project all ghost
    #           agents are always the same.
    #     """
    #
    #     newParticles = []
    #     for oldParticle in self.particles:
    #         newParticle = list(oldParticle)  # A list of ghost positions
    #         # now loop through and update each entry in newParticle...
    #         for i in self.getOpponents(gameState):
    #             # distribution for each particular enemy based on all enemy positions
    #             newPosDist = getPositionDistributionForGhost(
    #                 setGhostPositions(gameState, oldParticle), i, self.ghostAgents[i]
    #             )
    #             newParticle[i] = util.sample(newPosDist)
    #         newParticles.append(tuple(newParticle))
    #     self.particles = newParticles



################################################################################
# Things we don't really use below
################################################################################

class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    This is an abstract class, which you should not modify.
    """

    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        "Sets the ghost agent for later access"
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = [] # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistribution(self, gameState):
        """
        Returns a distribution over successor positions of the ghost from the
        given gameState.

        You must first place the ghost in the gameState, using setGhostPosition
        below.
        """
        ghostPosition = gameState.getGhostPosition(self.index) # The position you set
        actionDist = self.ghostAgent.getDistribution(gameState)
        dist = util.Counter()
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            dist[successorPosition] = prob
        return dist

    def setGhostPosition(self, gameState, ghostPosition):
        """
        Sets the position of the ghost for this inference module to the
        specified position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observeState.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[self.index] = game.AgentState(conf, False)
        return gameState

    def observeState(self, gameState):
        "Collects the relevant noisy distance observation and pass it along."
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index: # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observe(obs, gameState)

    def initialize(self, gameState):
        "Initializes beliefs to a uniform distribution over all positions."
        # The legal positions do not include the ghost prison cells in the bottom left.
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        "Sets the belief state to a uniform prior belief over all positions."
        pass

    def observe(self, observation, gameState):
        "Updates beliefs based on the given distance observation and gameState."
        pass

    def elapseTime(self, gameState):
        "Updates beliefs for a time step elapsing from a gameState."
        pass

    def getBeliefDistribution(self):
        """
        Returns the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        pass


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """

    def initializeUniformly(self, gameState):
        "Set the belief state to an initial, prior value."
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observeState(self, gameState):
        "Update beliefs based on the given distance observation and gameState."
        if self.index == 1:
            jointInference.observeState(gameState)

    # def elapseTime(self, gameState):
    #     "Update beliefs for a time step elapsing from a gameState."
    #     if self.index == 1:
    #         jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        "Returns the marginal belief over a particular ghost by summing out the others."
        jointDistribution = jointInference.getBeliefDistribution()
        dist = util.Counter()
        for t, prob in jointDistribution.items():
            # print(str(self.index))
            dist[t[self.index - 1]] += prob
        return dist

# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilterAgent()

def getPositionDistributionForGhost(gameState, ghostIndex, agent):
    """
    Returns the distribution over positions for a ghost, using the supplied
    gameState.
    """
    # index 0 is pacman, but the students think that index 0 is the first ghost.
    ghostPosition = gameState.getGhostPosition(ghostIndex+1)
    actionDist = agent.getDistribution(gameState)
    dist = util.Counter()
    for action, prob in actionDist.items():
        successorPosition = game.Actions.getSuccessor(ghostPosition, action)
        dist[successorPosition] = prob
    return dist

def setGhostPositions(gameState, ghostPositions):
    "Sets the position of all ghosts to the values in ghostPositionTuple."
    for index, pos in enumerate(ghostPositions):
        conf = game.Configuration(pos, game.Directions.STOP)
        gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
    return gameState
