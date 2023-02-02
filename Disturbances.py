# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 19:02:56 2022

@author: 253364
"""

import random

    
def random_noise(sensor_info, level): #creates random noise
    noise = []      #create the noise and output
    output = []
    for i in range(len(sensor_info)):                   #loop through the sensor information
        noise.append(random.uniform(-level, level))     #create noise
        
    for i in range(len(sensor_info)):                   #add the noise to sensors
        output = noise + sensor_info
        
    return output
        

class Impluse:
    
    def __init__(self):
        self.x = 0
        self.output = [0, 0, 0, 0, 0, 0, 0, 0] #set the x and the output list
    
    def implus(self, sensor_info, pulse):       #impulse
        self.x = self.x + 1                     #add one to x
        
        if self.x == pulse:                     #if iteration of pulse is triggered then add 1 to each sensor
            self.output = sensor_info + 1
            self.x = 0
        
        return self.output
    
def sensory_inversion(sensors):
    return sensors[::-1]            #flip the list

class sensory_inversion_pulse:
    
    def __init__(self):
        self.x = 0                      
        
    def pulse(self,sensors, pulse):
        self.x = self.x + 1
        
        if self.x == pulse:         #if pulse then flip the sensors
            self.x = 0
            return sensors[::-1]
        else:
            return sensors
    