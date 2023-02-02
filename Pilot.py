# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 19:31:12 2022

@author: 253364
"""
import random
from LanderV4 import Lander

class Pilot():
    
    def __init__(self, state_size, gym):
        
        self.state_size = state_size
        self.net1 = Lander(gamma = 0.99, epsilon= 1.0, learn_rate=0.01, input_size= [state_size], batch=64, output_size=4, input_layer_size= 256, hidden_layer_size= 256)
        self.net2 = Lander(gamma = 0.99, epsilon= 1.0, learn_rate=0.01, input_size= [state_size], batch=64, output_size=4, input_layer_size= 256, hidden_layer_size= 256)
        self.net3 = Lander(gamma = 0.99, epsilon= 1.0, learn_rate=0.01, input_size= [state_size], batch=64, output_size=4, input_layer_size= 256, hidden_layer_size= 256)
        self.net4 = Lander(gamma = 0.99, epsilon= 1.0, learn_rate=0.01, input_size= [state_size], batch=64, output_size=4, input_layer_size= 256, hidden_layer_size= 256)
        
        self.nets = [self.net1, self.net2, self.net3, self.net4]    #add the Landers
        self.actions = []                                           #set the senors
        
        self.gym = gym          #what enviroment is used
        
        
    def decisions(self, observation):
        self.actions = []
        for it in range(len(self.nets)):                                #Loop through all the networks
            self.actions.append(self.nets[it].decision(observation))    #make decision of the network
            
        if self.gym == "lunar":                                 #if lunar
            act = max(self.actions, key = self.actions.count)   #find the most common result
            try:
                if len(act) > 1:                        #if more then 1 most coomon randomly pick one
                    return random.choice(act)
            except:
                return act
            
        elif self.gym == "walker":
            return self.actions
        
        else:
            print("Please select a gym")
        
        

        
    def manage(self, observation, action, reward, state, done):
        
        for it in range(len(self.nets)):
            self.nets[it].transition(observation, action, reward, state, done) #update the memory in model
            self.nets[it].learn()                                               #backprogate through the DQN
            
        