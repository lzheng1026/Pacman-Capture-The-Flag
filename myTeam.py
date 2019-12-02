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
               first='DummyAgent', second='DummyAgent'):
    first = "OffensiveReflexAgent"
    second = "DummyAgent"
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########


class ParticlesCTFAgent(CaptureAgent):
    """
    CTF Agent that models enemies using particle filtering.
    """

    def registerInitialState(self, gameState, numParticles=600):
        # =====original register initial state=======
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        # =====ParticleCTFAgent init================
        self.setNumParticles(numParticles)
        self.initialize(gameState)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initialize(self, gameState, legalPositions=None):

        self.numEnemies = gameState.getNumAgents() - 2
        self.enemies = []
        self.legalPositions = gameState.getWalls().asList(False)
        self.initializeParticles()
        self.a, self.b = self.getOpponents(gameState)
        # for fail
        self.initialGameState = gameState
        # for features
        self.scaredMovesLeft = 0
        self.capsulesCount = len(self.getCapsules(gameState))

    def setEnemyPosition(self, gameState, pos, enemyIndex):

        conf = game.Configuration(pos, game.Directions.STOP)
        gameState.data.agentStates[enemyIndex] = game.AgentState(conf, False)
        return gameState

    def initializeParticles(self, type="both"):

        positions = self.legalPositions
        atEach = self.numParticles / len(positions)  # self.numParticles
        remainder = self.numParticles % len(positions)
        # don't throw out a particle
        particles = []
        # populate particles
        for pos in positions:
            for num in range(atEach):
                particles.append(pos)
        # now populate the remainders
        for index in range(remainder):
            particles.append(positions[index])
        # save to self.particles
        if type == 'both':
            self.particlesA = particles
            self.particlesB = particles
        elif type == self.a:
            self.particlesA = particles
        elif type == self.b:
            self.particlesB = particles
        return particles

    def observeState(self, gameState, enemyIndex):

        pacmanPosition = gameState.getAgentPosition(self.index)

        if enemyIndex == self.a:
            noisyDistance = gameState.getAgentDistances()[self.a]
            beliefDist = self.getBeliefDistribution(self.a)
            particles = self.particlesA
            if gameState.getAgentPosition(self.a) != None:
                self.particlesA = [gameState.getAgentPosition(self.a)] * self.numParticles
                return
        else:
            noisyDistance = gameState.getAgentDistances()[self.b]
            beliefDist = self.getBeliefDistribution(self.b)
            particles = self.particlesB
            if gameState.getAgentPosition(self.b) != None:
                self.particlesB = [gameState.getAgentPosition(self.b)] * self.numParticles
                return

        W = util.Counter()

        for p in particles:
            trueDistance = self.getMazeDistance(p, pacmanPosition)
            W[p] = beliefDist[p] * gameState.getDistanceProb(trueDistance, noisyDistance)

        # we resample after we get weights for each ghost
        if W.totalCount() == 0:
            particles = self.initializeParticles(enemyIndex)
        else:
            values = []
            keys = []
            for key, value in W.items():
                keys.append(key)
                values.append(value)

            if enemyIndex == self.a:
                self.particlesA = util.nSample(values, keys, self.numParticles)
            else:
                self.particlesB = util.nSample(values, keys, self.numParticles)

    def getBeliefDistribution(self, enemyIndex):
        allPossible = util.Counter()
        if enemyIndex == self.a:
            for pos in self.particlesA:
                allPossible[pos] += 1
        else:
            for pos in self.particlesB:
                allPossible[pos] += 1
        allPossible.normalize()
        return allPossible

    def getEnemyPositions(self, enemyIndex):
        """
        Uses getBeliefDistribution to predict where the two enemies are most likely to be
        :return: two tuples of enemy positions
        """
        return self.getBeliefDistribution(enemyIndex).argMax()

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        debug = False
        if action is None:
            successor = gameState
        else:
            successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if debug:
            print("pos " + str(pos))
            print("nearestPoint " + str(nearestPoint(pos)) + "\n")
            # The point of the if-else statement below is to make sure you are on a grid, because you could be between points
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

        # debug
        for feature in weights.keys():
            print(str(feature) + " " + str(features[feature]))
        print("\n")

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

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        # ============================================
        start = time.time()
        pacmanPosition = gameState.getAgentPosition(self.index)
        self.observeState(gameState, self.a)
        self.observeState(gameState, self.b)
        beliefs = [self.getBeliefDistribution(self.a), self.getBeliefDistribution(self.b)]
        self.displayDistributionsOverPositions(beliefs)
        # ============================================
        actions = gameState.getLegalActions(self.index)

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
            return bestAction  # chooses action that make you closest to your start state

        aPosition = self.getEnemyPositions(self.a)
        hypotheticalState = self.setEnemyPosition(gameState, aPosition, self.a)

        bPosition = self.getEnemyPositions(self.b)
        hypotheticalState = self.setEnemyPosition(hypotheticalState, bPosition, self.b)

        partner = self.partnerIndex(gameState)

        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
        return random.choice(bestActions)
        """
        if self.getMazeDistance(aPosition,pacmanPosition) < 10:
            order = [self.index, self.a]
            result = self.maxValue(hypotheticalState, order, 0, 2, -10000000, 10000000)
            #print("Result")
            #print(result)
            #print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
            return result[1]
        elif self.getMazeDistance(bPosition,pacmanPosition) < 10:
            order = [self.index, self.b]
            result = self.maxValue(hypotheticalState, order, 0, 2, -10000000, 10000000)
            #print("Result")
            #print(result)
            #print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
            return result[1]

        else:
            values = [self.evaluate(gameState, a) for a in actions]
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            #print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
            return random.choice(bestActions)
        """

    def maxValue(self, gameState, order, index, depth, alpha, beta):
        # returns a value and an action so getAction can return the best action
        if gameState.isOver() or depth == 0:
            return [self.evaluate(gameState, None), None]
        v = -10000000
        action = None
        for a in gameState.getLegalActions(order[0]):
            newState = gameState.generateSuccessor(order[0], a)
            newScore = self.minValue(newState, order, index + 1, depth, alpha, beta)
            if newScore > v:
                v = newScore
                action = a
            if v > beta:
                return [v, a]
            alpha = max(alpha, v)
        return [v, action]

    def minValue(self, gameState, order, index, depth, alpha, beta):
        if gameState.isOver() or depth == 0:
            return self.evaluate(gameState, None)
        v = 10000000
        for a in gameState.getLegalActions(order[index]):
            newState = gameState.generateSuccessor(order[index], a)
            # if pacman goes next, here is where depth is decremented
            if index + 1 >= len(order):
                v = min(v, self.maxValue(newState, order, 0, depth - 1, alpha, beta)[0])
            # if another enemy goes
            else:
                #change to max and [0] if using partner
                v = min(v, self.minValue(newState, order, index + 1, depth, alpha, beta))
            if v < alpha:
                return v
            beta = min(beta, v)
        return v

    def partnerIndex(self, gameState):
        redTeam = gameState.getRedTeamIndices()
        blueTeam = gameState.getBlueTeamIndices()
        if self.index in redTeam:
            redTeam.remove(self.index)
            return redTeam[0]
        else:
            blueTeam.remove(self.index)
            return blueTeam[0]


class OffensiveReflexAgent(ParticlesCTFAgent):

    def getFeaturesFood(self, gameState, action, features, numCarryingLimit=25):

        successor = self.getSuccessor(gameState, action)
        myPos = successor.getAgentState(self.index).getPosition()
        foodList = self.getFood(successor).asList()
        numFoodEaten = gameState.getAgentState(self.index).numCarrying

        #print("numFoodEaten " + str(numFoodEaten))

        if len(foodList) <= 2 or numFoodEaten >= numCarryingLimit:
            # we don't care about getting more food
            # only cares about going back & avoiding enemy

            # distance to any of the three points in the middle
            # top
            top = 0
            # middle
            middle = int((gameState.data.layout.height-1)/2)
            # bottom
            bottom = gameState.data.layout.height-1

            # print("height of layout  " + str(gameState.data.layout.height))
            # print("top " + str(top))
            # print("middle " + str(middle))
            # print("bottom " + str(bottom))

            # incentivize to go to closest of three,
            middleColumn = int((gameState.data.layout.width-1)/2)
            # print("middleColumn " + str(middleColumn))
            threeDistances = [nearestPoint((float(top), float(middleColumn))), nearestPoint((float(middle), float(middleColumn))), nearestPoint((float(bottom), float(middleColumn)))]
            # minToHome = min([self.getMazeDistance(myPos, position) for position in threeDistances])
            minToHome = self.getMazeDistance(myPos, self.start)
            # print("min to home " + str(minToHome))
            if minToHome == 0:
                minToHome = 0.000001
            # add to features
            features['distanceToHome'] = -float(minToHome)

        else:
            # we care about getting more food & avoiding enemy
            # don't care about going home

            # total number of food
            features['foodScore'] = -len(foodList)

            # Compute distance to the nearest food
            if len(foodList) > 0:  # This should always be True,  but better safe than sorry
                minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
                features['distanceToFood'] = minDistance

    def getFeatures(self, gameState, action):

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myPos = successor.getAgentState(self.index).getPosition()

        # food
        self.getFeaturesFood(gameState, action, features)

        # capsules
        capsuleList = self.getCapsules(gameState)
        if len(capsuleList) > 0:
            minCapDistance = min([self.getMazeDistance(myPos, capsule) for capsule in capsuleList])
            if minCapDistance < 5: # CHANGED
                features['distanceToCapsule'] = -minCapDistance
            else:
                features['distanceToCapsule'] = -8

        # scared time
        opponent_a, opponent_b = self.getOpponents(gameState)
        print("opponent a " + str(opponent_a))
        print("state of opponent a " + str(gameState.getAgentState(opponent_a)))
        print("scared timer of a " + str(gameState.getAgentState(opponent_a).scaredTimer))
        if gameState.getAgentState(opponent_a).scaredTimer > gameState.getAgentState(opponent_b).scaredTimer:
            scaredTime = gameState.getAgentState(opponent_a).scaredTimer
        else:
            scaredTime = gameState.getAgentState(opponent_b).scaredTimer
        print("inside features: scared time " + str(scaredTime))

        # enemies

        # do I care about the enemy?
        care = False
        halfway = gameState.data.layout.width/2
        if self.red:
            if myPos[0] > halfway:
                # I am on the enemy's side
                care = True
        else: # I am on the blue team
            if myPos[0] < halfway:
                # now i am on red team's side
                care = True
        #print("care? " + str(care) + "\n")

        if care:

            enemies = [successor.getAgentPosition(i) for i in self.getOpponents(successor)]
            #print(str(enemies[0]))
            #print(str(enemies[1]))
            enemy_one_pos = enemies[0]
            enemy_two_pos = enemies[1]
            min_enemy_dist = 99999999999
            if enemy_one_pos is not None:
                min_enemy_dist = min(min_enemy_dist, self.getMazeDistance(myPos, enemy_one_pos))
            if enemy_two_pos is not None:
                min_enemy_dist = min(min_enemy_dist, self.getMazeDistance(myPos, enemy_two_pos))
            if enemy_one_pos is None and enemy_two_pos is None:
                beliefs = [self.getBeliefDistribution(enemy_index) for enemy_index in self.getOpponents(gameState)]
                enemy_one_prob = beliefs[0][beliefs[0].argMax()]
                enemy_two_prob = beliefs[1][beliefs[1].argMax()]
                if enemy_one_prob > enemy_two_prob:
                    general_enemy_dist = self.getMazeDistance(myPos, beliefs[0].argMax())
                else:
                    general_enemy_dist = self.getMazeDistance(myPos, beliefs[1].argMax())
                if general_enemy_dist < 10:
                    features['generalEnemyDist'] = 1/float(general_enemy_dist)
                else:
                    features['generalEnemyDist'] = 0

            if self.scaredMovesLeft > 5:
                # either eat the enemy if you are close or don't care
                if min_enemy_dist != 99999999999:
                    features['eatEnemyDist'] = min_enemy_dist
                else:
                    features['eatEnemyDist'] = 0
            else:
                #print("here all the time")
                # be scared if they are very close; within 5
                if min_enemy_dist != 99999999999:
                    features['minEnemyDist'] = 1/float(min_enemy_dist)
                else:
                    features['minEnemyDist'] = 0

            # care about it a little if it is close enough?

        # scared moves
        if self.scaredMovesLeft > 0:
            self.scaredMovesLeft -= 1

        return features

    def getWeights(self, gameState, action):
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        return {'foodScore': 100, 'distanceToFood': -1, 'distanceToHome': 1000, 'distanceToCapsule': 1.2}
        #, 'minEnemyDist': -10000, 'eatEnemyDist': -1,, 'generalEnemyDist': -1000}


class DefensiveReflexAgent(ParticlesCTFAgent):

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

        #=======================================

        # Compute expected distance to invaders using noisy distances
        # offensive, we are actually scared of enemies
        enemyIndices = self.getOpponents(gameState)
        a = enemyIndices[0]
        b = enemyIndices[1]

        #expected dist
        a_distribution = self.getBeliefDistribution(a)
        a_dist = 0
        for loc, prob in a_distribution.items():
            a_dist += prob * self.getMazeDistance(loc, myPos)
        if a_dist < 20:
            features["enemy_a"] = 0
        else:
            features["enemy_a"] = a_dist

        b_distribution = self.getBeliefDistribution(b)
        b_dist = 0
        for loc, prob in b_distribution.items():
            b_dist += prob * self.getMazeDistance(loc, myPos)
        if b_dist < 20:
            features["enemy_b"] = 0
        else:
            features["enemy_b"] = a_dist

        if features["enemy_b"]<features["enemy_a"]:
            temp=features["enemy_a"]
            features["enemy_a"] = features["enemy_b"]
            features["enemy_b"] = temp

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2,
                "enemy_a": -100, "enemy_b": -10}



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