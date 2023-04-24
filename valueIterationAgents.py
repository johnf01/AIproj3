# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #print("\nFUNCTION CALLED: runValueIteration")
        self.values = util.Counter()
        #print(self.values)
        states = self.mdp.getStates()
        discount = self.discount
        iterations = self.iterations

        # print(states)
        # print(self.iterations)
        
        for k in range(iterations):
            #print("\n")
            #print("\n")
            #print("------------------KVALUE: ",k)
            #print(self.values)
            tempvalues = util.Counter()
            for state in states:
                #print("\n")
                #print("STATE: ",state)
                actions = self.mdp.getPossibleActions(state)
                # print(actions)
                max = -99999
                for action in actions:
                    #print("ACTION: ", action)
                    transitionStatesProbs = self.mdp.getTransitionStatesAndProbs(state, action)
                    #print(transitionStatesProbs)
                    sum = 0.0
                    for t in transitionStatesProbs:
                        T = t[1] 
                        #print(t[0])
                        R = self.mdp.getReward(state, action, t[0])
                        l =  self.discount 
                        Vk = self.values[t[0]]
                        #print("---T: ",T," R: ", R,"discount: ", l, "Vk: ", Vk)
                        sum += T * (R + l*Vk)
                        #print("-----------SUM: ",sum)
                    if sum > max:
                        max = sum
                        #print("-----------MAX: ",max)
                
                # print("-----state before update: ", self.values[state])
                if(max != -99999):
                    tempvalues[state]=max
            for state in states:
                self.values[state] = tempvalues[state]
                

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # print("\nFUNCTION CALLED: computeQValuesFromValues")
        
        stateProb = self.mdp.getTransitionStatesAndProbs(state, action)
        sum = 0.0
        for t in stateProb:
            T = t[1] 
            R = self.mdp.getReward(state, action, t[0])
            l =  self.discount
            Vk = self.values[t[0]]
            sum += T * (R + l*Vk)
        return sum

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # print("\nFUNCTION CALLED: computeActionFromValues")
        # print(self.values)
        # print("\nSTATE: ",state)
        
        if state == 'TERMINAL_STATE':
            return None
        
        policy = 0.0
        bestAction = None
        max = -9999
        
        for action in self.mdp.getPossibleActions(state):
            # print("ACTION: ",action)
            #newState = self.mdp.getTransitionStatesAndProbs(state, action)
            policy = self.computeQValueFromValues(state, action)
            # print("newState: ", newState)
            # print("newState coords: ",newState[0])
            # policy = self.values[newState[0]]
            # print("POLICY: ",policy)
            if(policy > max):
                max = policy
                bestAction = action
        return bestAction

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

