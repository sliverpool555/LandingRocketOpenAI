# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 19:57:00 2022

@author: 253364
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

from Pilot import Pilot
from functions import save_agent, load_agent
from Disturbances import random_noise, Impluse
from Logger import Logger





env = gym.make('LunarLander-v2')    #create the Lunar lander enviroment
pilot = Pilot(8, "lunar")           #Create the Pilot

scores = []         #create list to store scores
avg_scores = []

amount = 400                #Set the amount of realisations

best_score = -10000         #set initial scores really low
best_avg_score = -10000

logger = Logger()   #create the Logger object to store data

implus = Impluse()  #Create the impluse

for i in range(amount):
    score = 0                   #set score to 0
    done = False                #reset the done to False
    observation = env.reset()   #Set the obervation from the sensors
    while not done:
        action = pilot.decisions(observation)           #Parse through the sensor information to the pilots decision to make an action
        state, reward, done, info = env.step(action)    #Update enviroment
    
        #observation = random_noise(observation)
        #observation = implus.implus(observation, 10)
        
        score = score + reward                                  #update the score                                       
        pilot.manage(observation, action, reward, state, done)  #Pilot to learn
        
        observation = state     #set the state to obversation for feedback
        
    scores.append(score)        #update the scores
    
    avg_score = np.mean(scores)     #find the mean of scores
    avg_scores.append(avg_score)    #Add average scores to lists
    
    logger.log(eps=pilot.net1.epsilon, score=score, avg=avg_score, sensors=observation)     #log the results for realisation
    
    print("Episode: ", i, "Score: ", score, "Average Score: ", avg_score, "Epsilion ", pilot.net1.epsilon)
    
    if avg_score > best_avg_score:                  #if the score is greater then the best score then save this pilot
        print("Best model    Score: ", avg_score)
        print()
        save_agent(pilot, "score_walker.pickle")
        best_avg_score = avg_score
        
    if score > best_score:                          #if the score is greater then the best score then save this pilot
        print("Best Model for score")
        print()
        save_agent(pilot, "best_score_walker.pickle")
        best_score = score
        

logger.plot_avg()           #plot the data
logger.plot_score()
logger.plot_learning()
    

save_agent(pilot, "Last_agent.pickle") #print final result