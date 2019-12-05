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

debug = False


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

        # =====ParticleCTFAgent init================
        self.numParticles = numParticles
        self.initialize(gameState)
        # =====Features=============
        self.numFoodToEat = len(self.getFood(gameState).asList())-2
        self.scaredMoves = 0

        CaptureAgent.registerInitialState(self, gameState)

    def initialize(self, gameState, legalPositions=None):
        self.legalPositions = gameState.getWalls().asList(False)
        self.initializeParticles()
        self.a, self.b = self.getOpponents(gameState)
        # for fail
        self.initialGameState = gameState

    def setEnemyPosition(self, gameState, pos, enemyIndex):
        foodGrid = self.getFood(gameState)
        halfway = foodGrid.width/2
        conf = game.Configuration(pos, game.Directions.STOP)

        #FOR THE WEIRD ERROR CHECK
        if gameState.isOnRedTeam(self.index):
            if pos[0] >= halfway:
                isPacman = False
            else:
                isPacman = True
        else:
            if pos[0] >= halfway:
                isPacman = True
            else:
                isPacman = False
        gameState.data.agentStates[enemyIndex] = game.AgentState(conf, isPacman)


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
        if debug:
            for feature in weights.keys():
                print(str(feature) + " " + str(features[feature]) + "; feature weight: " + str(weights[feature]))
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

        start = time.time()
        pacmanPosition = gameState.getAgentPosition(self.index)
        self.observeState(gameState, self.a)
        self.observeState(gameState, self.b)
        beliefs = [self.getBeliefDistribution(self.a), self.getBeliefDistribution(self.b)]
        #self.displayDistributionsOverPositions(beliefs)

        actions = gameState.getLegalActions(self.index)

        aPosition = self.getEnemyPositions(self.a)
        hypotheticalState = gameState.deepCopy()
        hypotheticalState = self.setEnemyPosition(hypotheticalState, aPosition, self.a)

        bPosition = self.getEnemyPositions(self.b)
        hypotheticalState = self.setEnemyPosition(hypotheticalState, bPosition, self.b)

        if self.getMazeDistance(aPosition, pacmanPosition) < 7 and self.getBeliefDistribution(self.a)[aPosition] > 0.5:
            #print("***** in mini max ******")
            order = [self.index, self.a]
            if self.getMazeDistance(aPosition, pacmanPosition) < 3:
                result = self.maxValue(hypotheticalState, order, 0, 3, -10000000, 10000000,start)
            else:
                result = self.maxValue(hypotheticalState, order, 0, 2, -10000000, 10000000, start)
            # update scared moves
            if len(self.getCapsules(gameState)) != len(self.getCapsules(self.getSuccessor(gameState, result[1]))):
                # we ate a capsule!
                self.scaredMoves = self.scaredMoves + 40
            elif self.scaredMoves != 0:
                self.scaredMoves = self.scaredMoves - 1
            else:
                pass
            #if time.time() - start > 0.8:
                #print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
            return result[1]
        elif self.getMazeDistance(bPosition, pacmanPosition) < 7 and self.getBeliefDistribution(self.b)[bPosition] > 0.5:
            #print("***** in mini max ******")
            order = [self.index, self.b]
            if self.getMazeDistance(bPosition, pacmanPosition) < 3:
                result = self.maxValue(hypotheticalState, order, 0, 3, -10000000, 10000000, start)
            else:
                result = self.maxValue(hypotheticalState, order, 0, 2, -10000000, 10000000, start)
            # update scared moves
            if len(self.getCapsules(gameState)) != len(self.getCapsules(self.getSuccessor(gameState, result[1]))):
                # we ate a capsule!
                self.scaredMoves = self.scaredMoves + 40
            elif self.scaredMoves != 0:
                self.scaredMoves = self.scaredMoves - 1
            else:
                pass
            #if time.time() - start > 0.8:
                #print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
            return result[1]

        else:
            values = [self.evaluate(gameState, a) for a in actions]
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            bestAction = random.choice(bestActions)
            # update scared moves
            if len(self.getCapsules(gameState)) != len(self.getCapsules(self.getSuccessor(gameState, bestAction))):
                # we ate a capsule!
                self.scaredMoves = self.scaredMoves + 40
            elif self.scaredMoves != 0:
                self.scaredMoves = self.scaredMoves - 1
            else:
                pass
            #print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
            return bestAction


    def maxValue(self, gameState, order, index, depth, alpha, beta, start):
        # returns a value and an action so getAction can return the best action
        if gameState.isOver() or depth == 0 or ((time.time()-start) > 0.9):
            return [self.evaluate(gameState, None), None]
        v = -10000000
        action = None
        for a in gameState.getLegalActions(order[0]):
            try:
                newState = gameState.generateSuccessor(order[0], a)
            except:
                print("exception occured")
                return [self.evaluate(gameState, None), None]
            newScore = self.minValue(newState, order, index + 1, depth, alpha, beta,start)
            if newScore > v:
                v = newScore
                action = a
            if v > beta:
                return [v, a]
            alpha = max(alpha, v)
        return [v, action]

    def minValue(self, gameState, order, index, depth, alpha, beta, start):
        if gameState.isOver() or depth == 0 or ((time.time()-start) > 0.9):
            return self.evaluate(gameState, None)
        v = 10000000
        for a in gameState.getLegalActions(order[index]):
            try:
                newState = gameState.generateSuccessor(order[index], a)
            except:
                print("exception occured")
                return self.evaluate(gameState, None)
            # if pacman goes next, here is where depth is decremented
            if index + 1 >= len(order):
                v = min(v, self.maxValue(newState, order, 0, depth - 1, alpha, beta,start)[0])
            # if another enemy goes
            else:
                #change to max and [0] if using partner
                v = min(v, self.minValue(newState, order, index + 1, depth, alpha, beta,start))
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

    def getFeaturesFood(self, myPos, numFoodEaten, foodList, shouldGoHome, features, numCarryingLimit):

        if len(foodList) <= 2 or (numFoodEaten>=numCarryingLimit and shouldGoHome):
            # we don't care about getting more food
            # only cares about going back & avoiding enemy

            minToHome = self.getMazeDistance(myPos, self.start)

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

        foodGrid = self.getFood(successor)
        foodList = self.getFood(successor).asList()
        numCarryingLimit = int(self.numFoodToEat / 3)
        numFoodEaten = gameState.getAgentState(self.index).numCarrying

        width = foodGrid.width
        height = foodGrid.height
        halfway = foodGrid.width / 2

        shouldGoHome = False
        if features['minEnemyDist'] > 0 or abs(myPos[0] - halfway) < 3:
            shouldGoHome = True

        # food
        self.getFeaturesFood(myPos, numFoodEaten, foodList, shouldGoHome, features, numCarryingLimit)

        # capsules
        capsuleList = self.getCapsules(gameState)
        if len(capsuleList) > 0:
            minCapDistance = min([self.getMazeDistance(myPos, capsule) for capsule in capsuleList])
            if minCapDistance < 5: # CHANGED
                features['distanceToCapsule'] = -minCapDistance
            else:
                features['distanceToCapsule'] = -8

        # scared time
        localScaredMoves = 0
        # when we eat a capsule
        if len(capsuleList) != len(self.getCapsules(successor)):
            # we ate a capsule!
            localScaredMoves = self.scaredMoves + 40
        elif self.scaredMoves != 0:
            localScaredMoves = self.scaredMoves-1

        # enemies

        halfway = width/2
        when_gen_enemy_dist_matters = int(min(width, height) * 2 / 3)
        when_min_enemy_dist_matters = 10

        if gameState.getAgentState(self.index).isPacman:

            enemies = self.getOpponents(successor)
            enemy_one_pos = successor.getAgentPosition(enemies[0])
            enemy_two_pos = successor.getAgentPosition(enemies[1])
            min_enemy_dist = 99999999999

            # if enemies are in viewing
            if enemy_one_pos is not None and ((enemy_one_pos[0] > halfway and myPos[0] > halfway) or (enemy_one_pos[0] < halfway and myPos[0] < halfway)):
                min_enemy_dist = min(min_enemy_dist, self.getMazeDistance(myPos, enemy_one_pos))

            if enemy_two_pos is not None and ((enemy_two_pos[0] > halfway and myPos[0] > halfway) or (enemy_two_pos[0] < halfway and myPos[0] < halfway)):
                min_enemy_dist = min(min_enemy_dist, self.getMazeDistance(myPos, enemy_two_pos))

            # if enemy is not in viewing
            if enemy_one_pos is None and enemy_two_pos is None:
                beliefs = [self.getBeliefDistribution(enemy_index) for enemy_index in self.getOpponents(gameState)]
                enemy_one_loc = beliefs[0].argMax()
                enemy_two_loc = beliefs[1].argMax()
                enemy_one_dist = self.getMazeDistance(myPos, enemy_one_loc)
                enemy_two_dist = self.getMazeDistance(myPos, enemy_two_loc)
                general_enemy_dist = 99999999999
                if enemy_one_dist < general_enemy_dist and ((enemy_one_loc[0] > halfway and myPos[0] > halfway) or (enemy_one_loc[0] < halfway and myPos[0] < halfway)):
                    general_enemy_dist = enemy_one_dist
                if enemy_two_dist < general_enemy_dist and ((enemy_two_loc[0] > halfway and myPos[0] > halfway) or (enemy_two_loc[0] < halfway and myPos[0] < halfway)):
                    general_enemy_dist = enemy_two_dist

                if general_enemy_dist < when_gen_enemy_dist_matters: # CAN BE MODIFIED
                    features['generalEnemyDist'] = general_enemy_dist
                else:
                    features['generalEnemyDist'] = when_gen_enemy_dist_matters

            if localScaredMoves > 0:

                # either eat the enemy if you are close or don't care
                if min_enemy_dist != 99999999999 and min_enemy_dist < when_min_enemy_dist_matters:
                    # you are close!
                    features['eatEnemyDist'] = when_min_enemy_dist_matters-float(min_enemy_dist)
                else:
                    features['eatEnemyDist'] = 0
            else:
                # be scared if they are very close
                if min_enemy_dist != 99999999999 and min_enemy_dist < when_min_enemy_dist_matters: # could break if maze is really small
                    features['minEnemyDist'] = when_min_enemy_dist_matters-float(min_enemy_dist)
                else:
                    features['minEnemyDist'] = 0

        else:
            features['generalEnemyDist'] = 20

        return features

    def getWeights(self, gameState, action):

        return {'foodScore': 100, 'distanceToFood': -2, 'distanceToHome': 1000, 'distanceToCapsule': 1.2,
                'minEnemyDist': -100, 'generalEnemyDist': 1, 'eatEnemyDist': 2.1}


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