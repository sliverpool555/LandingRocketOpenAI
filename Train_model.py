# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 21:20:29 2022

@author: 253364
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

from LanderV4 import Lander
from functions import save_agent, load_agent
from Disturbances import random_noise, Impluse
from Logger import Logger





env = gym.make('LunarLander-v2')    #Create enviroment
agent = Lander(gamma = 0.99, epsilon= 1.0, learn_rate=0.01, input_size= [8], batch=64, output_size=4, input_layer_size= 256, hidden_layer_size= 256) #create the lander

scores = []         #create the scores
avg_scores = []

amount = 400        #set the amount of training epochs

best_score = -10000         #set the intial scores
best_avg_score = -10000

logger = Logger()   #create the logger

implus = Impluse()  #create the logger

for i in range(amount):
    score = 0                   #reset the socres
    done = False                #set the enviorment back to False to run again
    observation = env.reset()   #reset the enviorment sensory scores 
    while not done:
        action = agent.decision(observation)            #make a decisions
        state, reward, done, info = env.step(action)    #Move the agent
    
        #observation = random_noise(observation)
        #observation = implus.implus(observation, 10)
        
        score = score + reward                          #update the score
        agent.transition(observation, action, reward, state, done)  #add data to memory
        
        agent.learn() #Agent to learn from data
        observation = state #set the sensory iformation to the obversation
        
    scores.append(score)                #add the scores to the list

    
    avg_score = np.mean(scores)
    avg_scores.append(avg_score)        #add the average
    
    logger.log(eps=agent.epsilon, score=score, avg=avg_score, sensors=observation)  #Add data to the logger
    
    print("Episode: ", i, "Score: ", score, "Average Score: ", avg_score, "Epsilion ", agent.epsilon) #print the realisation
    
    if avg_score > best_avg_score:                  #if the model is better then the pervous then save the model
        print("Best model    Score: ", avg_score)
        print()
        save_agent(agent, "best_avg_LR.pickle")
        best_avg_score = avg_score
        
    if score > best_score:
        print("Best Model for score")
        print()
        save_agent(agent, "best_score_LR.pickle")
        best_score = score
        

logger.plot_avg()       #log the data
logger.plot_score()
logger.plot_learning()
    

save_agent(agent, "Last_agent.pickle") #save the last data
