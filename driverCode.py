# -*- coding: utf-8 -*-
"""
Reinforcement Learning Tutorial:

First Visit Monte Carlo Method for Learning the state value function 
for a given policy

Tested on the Frozen Lake OpenAI Gym environment.
Author: Aleksandar Haber 
Date: December 2022 

This is the driver code that coles functions from "functions.py"

"""

import gym
import numpy as np  

from functions import MonteCarloLearnStateValueFunction
from functions import evaluatePolicy
from functions import grid_print

# create an environment 

# note here that we create only a single hole to makes sure that we do not need
# a large number of simulations
# generate a custom Frozen Lake environment
desc=["SFFF", "FFFF", "FFFF", "HFFG"]

# here we render the environment- use this only for illustration purposes
# env=gym.make('FrozenLake-v1', desc=desc, map_name="4x4", is_slippery=True,render_mode="human")

# uncomment this and comment the previous line in the case of a large number of simulations
env=gym.make('FrozenLake-v1', desc=desc, map_name="4x4", is_slippery=True)

# number of states in the environment
stateNumber=env.observation_space.n

# number of simulation episodes
numberOfEpisodes=10000

# discount rate
discountRate=1
# estimate the state value function by using the Monte Carlo method
estimatedValuesMonteCarlo=MonteCarloLearnStateValueFunction(env,stateNumber=stateNumber,numberOfEpisodes=numberOfEpisodes,discountRate=discountRate)


# for comparison compute the state value function vector by using the iterative policy 
# evaluation algorithm

# select an initial policy
# initial policy starts with a completely random policy
# that is, in every state, there is an equal probability of choosing a particular action
initialPolicy=(1/4)*np.ones((16,4))

# initialize the value function vector
valueFunctionVectorInitial=np.zeros(env.observation_space.n)
# maximum number of iterations of the iterative policy evaluation algorithm
maxNumberOfIterationsOfIterativePolicyEvaluation=1000
# convergence tolerance 
convergenceToleranceIterativePolicyEvaluation=10**(-6)

# the iterative policy evaluation algorithm
valueFunctionIterativePolicyEvaluation=evaluatePolicy(env,valueFunctionVectorInitial,initialPolicy,1,maxNumberOfIterationsOfIterativePolicyEvaluation,convergenceToleranceIterativePolicyEvaluation)

# plot the result
grid_print(valueFunctionIterativePolicyEvaluation,reshapeDim=4,fileNameToSave='iterativePolicyEvaluationEstimated.png')

# plot the result
grid_print(estimatedValuesMonteCarlo,reshapeDim=4,fileNameToSave='monteCarloEstimated.png')
       
        