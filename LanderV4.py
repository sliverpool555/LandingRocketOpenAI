# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 14:53:22 2022


"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class DQN(nn.Module):
    
    def __init__(self, learn_rate, inputs, input_size, hidden_size, output_size):
        super(DQN, self).__init__()         #Inherit all methods from pytorches nn.Module
        self.inputs = inputs                #set the inputs
        self.input_size = input_size        #Set the input size
        self.hidden_size = hidden_size      #set the size of the hiddern layer
        self.output_size = output_size      #set the output size
        
        self.input_layer = nn.Linear(*self.inputs, self.input_size)         #create the input layer
        self.hidden_layer = nn.Linear(self.input_size, self.hidden_size)    #create the hidden layer
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)   #Create the output layer
        
        self.optimizer = optim.Adam(self.parameters(), lr = learn_rate)     #create the optimzier
        self.loss = nn.MSELoss()
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #Set the device 
        self.to(self.device)        #Make it easy to use device
        
        
    def forward(self, state):                       #Feedward function
        out = F.sigmoid(self.input_layer(state))    #Parse the input into the input layers
        out = F.sigmoid(self.hidden_layer(out))     #Take the output from the input layer to feedward prograte through hiddern
        out = self.output_layer(out)                #Output of hidden layer to parse through the output layer to find output of network
        
        return out
    
    
    
    
    
    
class Lander():
    
    def __init__(self, gamma, epsilon, learn_rate, input_size, batch, output_size, input_layer_size, hidden_layer_size):
        
        self.memory_size = 10000                            #set the size of memory
        self.epsiode = 0.01                                 #Set the 
        self.epsiode_constant = 5e-4
        self.learn_rate = learn_rate                        #Set the Learning rate
        self.actions = [i for i in range(output_size)]      #create the actions
        self.batch_size = batch                             #set the batch
        self.memory_count = 0                               #Set the memory count to 0
        
        self.gamma = gamma                                  #set the gamma
        self.epsilon = epsilon                              #Set the epsilon to 1.0 as it will decrease
        
        self.model = DQN(self.learn_rate, input_size, input_layer_size, hidden_layer_size, output_size) #Add the model
        
        self.memory = np.zeros((self.memory_size, *input_size), dtype=np.float32)               #Create the memory for the DQN Agent was made from Machine Learning with Phill (2020).
        self.input_state_memory = np.zeros((self.memory_size, *input_size), dtype=np.float32)   
        
        self.action_memory = np.zeros(self.memory_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.bool)
        
        
    def transition(self, state, action, reward, new_state, done):
        
        index = self.memory_count % self.memory_size                    #This memeory function is from Machine Learning with Phill (2020). 
        self.memory[index] = state
        self.input_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        
        self.memory_count = self.memory_count + 1
        
    def decision(self, new_state):
        
        if np.random.random() > self.epsilon:                           #This memeory function is from Machine Learning with Phill (2020). 
            state = torch.tensor([new_state]).to(self.model.device)
            actions = self.model.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.actions)
            
        return action
    
    def preprocess(self):
        self.max_mem = min(self.memory_count, self.memory_size)                         #This memeory function is from Machine Learning with Phill (2020). 
        self.batch = np.random.choice(self.max_mem, self.batch_size, replace=False)
        
        self.state_batch = torch.tensor(self.memory[self.batch]).to(self.model.device)
        self.new_state_batch = torch.tensor(self.input_state_memory[self.batch]).to(self.model.device)
        self.reward_batch = torch.tensor(self.reward_memory[self.batch]).to(self.model.device)
        self.terminal_batch = torch.tensor(self.terminal_memory[self.batch]).to(self.model.device)
        
        self.action_batch = self.action_memory[self.batch]
        
    
    
    def find_Q(self):
        self.batch_index = np.arange(self.batch_size, dtype=np.int32)                       #This memeory function is from Machine Learning with Phill (2020). 
        model = self.model.forward(self.state_batch)[self.batch_index, self.action_batch]
        q_next = self.model.forward(self.new_state_batch)
        q_next[self.terminal_batch] = 0.0
        q_target = self.reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        return q_target, model
    
    
    def learn(self):
        
        if self.memory_count < self.batch_size:     #If memory count is less then the batch size return
            return 
        
        self.model.optimizer.zero_grad()            #Optimizer the model
        
        self.preprocess()                           #preprocess the data and store the data for the DQN
        
        q_target, model = self.find_Q()             #Apply the policys on the Model
        
        loss = self.model.loss(q_target, model).to(self.model.device) #Find the loss
        loss.backward()                                                 #change the weights depending on the output of the model
        self.model.optimizer.step()                                     #Apply ADAM optimizer
        
        self.epsilon = self.epsilon - self.epsiode_constant if self.epsilon > self.epsiode \
            else self.epsiode   # This line is from Machine Learning with Phill (2020). 
        
        
        
    
"""
Machine Learning with Phill (2020). 
Deep Q Learning is Simple with PyTorch | Full Tutorial 2020. [online] Available at: https://www.youtube.com/watch?v=wc-FxNENg9U&t=1403s [Accessed 1 Apr. 2022]      
"""     
    
    
        
