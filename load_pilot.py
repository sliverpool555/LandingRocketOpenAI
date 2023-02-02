# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 21:28:32 2022

@author: 253364
"""



import gym
import numpy as np
import matplotlib.pyplot as plt

from Pilot import Pilot
from functions import save_agent, load_agent
from Logger import Logger
from Disturbances import random_noise, Impluse


loaded_agent = load_agent('More_learning_pilot.pickle')


env = gym.make('LunarLander-v2')
env.render()


score = 0
done = False
observation = env.reset()

logger = Logger()
imp = Impluse()

while not done:
    env.render()
    action = loaded_agent.decisions(observation)
    
    #observation = random_noise(observation)
    #observation = imp.impluse(observation, 5)
    
    state, reward, done, info = env.step(action)
    
    score += reward
    loaded_agent.manage(observation, action, reward, state, done)
    
    observation = state
    
    logger.log_loaded(observation)
    
print(score)

env.close()


#logger.plot_sensor_correlation(loaded_agent.sensor0, loaded_agent.sensor1)

logger.plot_sensor0()
logger.plot_sensor1()
logger.plot_sensor2()
logger.plot_sensor3()
logger.plot_sensor4()
logger.plot_sensor5()
logger.plot_sensor6()
logger.plot_sensor7()
