# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 00:45:36 2022

@author: 253364
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

from LanderV4 import Lander
from functions import save_agent, load_agent
from Logger import Logger
from Disturbances import random_noise, Impluse, sensory_inversion, sensory_inversion_pulse


#oaded_agent = [load_agent("landerAgent.pickle"), load_agent('best_avg.pickle'), load_agent('best_score.pickle'), load_agent('Last_agent.pickle')]

loaded_agent = load_agent('Activation_expirment\\best_avg_sigmoid.pickle')


env = gym.make('LunarLander-v2')
env.render()


score = 0
done = False
observation = env.reset()           #reset the enviroment

logger = Logger()       #create logger
imp = Impluse()         #create impulse
inversion = sensory_inversion_pulse()           #create the inversion object

while not done:
    env.render()
    action = loaded_agent.decision(observation)             #find the action from model
    
    observation = random_noise(observation, 0.1)            #add distrubances
    #observation = imp.impluse(observation, 5)
    #observation = sensory_inversion(observation)
    #observation = inversion.pulse(observation, 1)
    
    state, reward, done, info = env.step(action)
    
    score += reward
    loaded_agent.transition(observation, action, reward, state, done) #update the memory
    
    loaded_agent.learn()  #update the DQN model               
    observation = state
    
    logger.log_loaded(observation) #log the sensory information
    
print(score)

env.close()     #close the enviroment


#logger.plot_sensor_correlation(loaded_agent.sensor0, loaded_agent.sensor1)

logger.plot_sensor0() #Plot the sensory information
logger.plot_sensor1()
logger.plot_sensor2()
logger.plot_sensor3()
logger.plot_sensor4()
logger.plot_sensor5()
logger.plot_sensor6()
logger.plot_sensor7()
logger.plot_sensor8()
