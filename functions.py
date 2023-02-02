# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 00:15:54 2022

@author: 253364
"""


import torch as T
import matplotlib.pyplot as plt 
import numpy as np


def save_agent(state, filename):
    T.save(state, filename)
    
    
def load_agent(file_name):
    
    print("Loading Check point")
    
    model = T.load(file_name)
 
    return model
        
