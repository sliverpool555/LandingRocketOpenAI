# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 18:58:08 2022

@author: 253364
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

from Pilot import Pilot
from functions import save_agent, load_agent
from Logger import Logger
from Disturbances import sensory_inversion_pulse, sensory_inversion, Impluse


#oaded_agent = [load_agent("landerAgent.pickle"), load_agent('best_avg.pickle'), load_agent('best_score.pickle'), load_agent('Last_agent.pickle')]

#loaded_agent = load_agent('best_avg.pickle')

loaded_agent = load_agent("More_learning_pilot.pickle")

env = gym.make('LunarLander-v2')
env.render()


score = 0
done = False
observation = env.reset()

logger = Logger()


landed = 0
crashed = 0
results = []
iterations = []
labels = []

tests = 200

inversion = sensory_inversion_pulse()
imp = Impluse()

for n in range(tests): #iterate through the tests
    score = 0
    done = False
    observation = env.reset() #set sensory information
    while not done:
        #env.render()
        action = loaded_agent.decisions(observation) #find the action of the DQN
        state, reward, done, info = env.step(action) #NMove agent in enviroment
        
        #observation = imp.implus(observation, 50)
        #observation = sensory_inversion(observation)
        #observation = inversion.pulse(observation, 1)
        
        score = score + reward        #loaded_agent.transition(observation, action, reward, state, done)
        
        #loaded_agent.manage(observation, action, reward, state, done)
        observation = state
        
        logger.log_loaded(observation)
        
    print(n, score)
    
    env.close()

    if score > 0.0:                 #add one to the score if landed
        landed = landed + 1
        labels.append('Landed')
    else:
        crashed = crashed + 1       #add one if it is a crash
        labels.append('Crashed')
        
    iterations.append(n)
    results.append(score)
    
        
        
logger.plot_pie(landed, crashed)                    #plot the data
logger.plot_bar_chart(landed, crashed)
logger.plot_scatter(iterations, results, labels)


print("Averge Score: ", (sum(results)/len(results))) #print the average result of tests


save_agent(loaded_agent, "More_learning_pilot.pickle") #save the agent