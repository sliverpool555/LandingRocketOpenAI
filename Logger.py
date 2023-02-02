# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 20:34:16 2022

@author: Samuel Gandy
"""

import matplotlib.pyplot as plt


class Logger:
    
    def __init__(self):
        self.epsilion = []
        self.avg = []
        self.scores = []
        self.sensor0 = []
        self.sensor1 = []
        self.sensor2 = []
        self.sensor3 = []
        self.sensor4 = []
        self.sensor5 = []
        self.sensor6 = []
        self.sensor7 = []
        
        
        
    
    def log(self, eps, score, avg, sensors):
        self.epsilion.append(eps)
        self.scores.append(score)
        self.avg.append(avg)
        self.sensor0.append(sensors[0])
        self.sensor1.append(sensors[1])
        self.sensor2.append(sensors[2])
        self.sensor3.append(sensors[3])
        self.sensor4.append(sensors[4])
        self.sensor5.append(sensors[5])
        self.sensor6.append(sensors[6])
        self.sensor7.append(sensors[7])
        
    def log_loaded(self, sensors):
        self.sensor0.append(sensors[0])
        self.sensor1.append(sensors[1])
        self.sensor2.append(sensors[2])
        self.sensor3.append(sensors[3])
        self.sensor4.append(sensors[4])
        self.sensor5.append(sensors[5])
        self.sensor6.append(sensors[6])
        self.sensor7.append(sensors[7])
        
    
    def plot_learning(self):
        plt.title("Learning Graph")
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.grid()
        plt.plot(self.epsilion)
        plt.show()
    
    
    def plot_avg(self):
        plt.title("Iterations to Averages")
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.grid()
        plt.plot(self.avg)
        plt.show()
    
    
    def plot_score(self):
        plt.title("Iterations to scores")
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.grid()
        plt.plot(self.scores)
        plt.show()
        
        
    def plot_sensor_correlation(self, sensor_data1, sensor_data2):
        plt.title("Sensor Information")
        plt.xlabel("Iteration")
        plt.ylabel("Sensor info")
        plt.grid()
        plt.plot(sensor_data1)
        plt.plot(sensor_data2)
        plt.show()
        
        
    def plot_sensor0(self):
        plt.title("Sensor Information")
        plt.xlabel("Iteration")
        plt.ylabel("Sensor info")
        plt.grid()
        plt.plot(self.sensor0, self.sensor1)
        plt.show()
    
    def plot_sensor1(self):
        plt.title("X position")
        plt.xlabel("Iteration")
        plt.ylabel("X coordinate")
        plt.grid()
        plt.plot(self.sensor0)
        plt.show()
    
    def plot_sensor2(self):
        plt.title("Y Position")
        plt.xlabel("Iteration")
        plt.ylabel("Y coordinate")
        plt.grid()
        plt.plot(self.sensor1)
        plt.show()
    
    def plot_sensor3(self):
        plt.title("Right Velocity")
        plt.xlabel("Iteration")
        plt.ylabel("Sensor info")
        plt.grid()
        plt.plot(self.sensor2)
        plt.show()
    
    def plot_sensor4(self):
        plt.title("Left Velocity")
        plt.xlabel("Iteration")
        plt.ylabel("Sensor info")
        plt.grid()
        plt.plot(self.sensor3)
        plt.show()
    
    def plot_sensor5(self):
        plt.title("Right Angular Velocity")
        plt.xlabel("Iteration")
        plt.ylabel("Sensor info")
        plt.grid()
        plt.plot(self.sensor4)
        plt.show()
    
    def plot_sensor6(self):
        plt.title("Left Angular Velocity")
        plt.xlabel("Iteration")
        plt.ylabel("Sensor info")
        plt.grid()
        plt.plot(self.sensor5)
        plt.show()
    
    def plot_sensor7(self):
        plt.title("Right Leg Sensor")
        plt.xlabel("Iteration")
        plt.ylabel("Sensor info")
        plt.grid()
        plt.plot(self.sensor6)
        plt.show()
        
    def plot_sensor8(self):
        plt.title("Left Leg Sensor")
        plt.xlabel("Iteration")
        plt.ylabel("Sensor info")
        plt.grid()
        plt.plot(self.sensor7)
        plt.show()
        

    
    def plot_bar_chart(self, land, crash):
        
        labels = ['landed', 'crashed']
        
        data = [land, crash]
        
        stand_label = [i for i, _ in enumerate(labels)]
        
        ratio = land/crash
        
        plt.bar(stand_label, data, color='gray')
        plt.xlabel('Amount')
        plt.ylabel('Crashed to Landed')
        plt.title("Landing to Crached ratio {:.2f}". format(ratio))
        plt.xticks(stand_label, labels)
        plt.show()
        
    
    def plot_pie(self, land, crash):
        
        ratio = land/crash
        
        labels = ['landed', 'crashed']
        
        data = [land, crash]
        
        plt.pie(data, labels=labels)
        plt.title("Landing to Crached ratio {:.2f}". format(ratio))
        plt.show()
        
    
    def plot_scatter(self, iterations, data, labels):
        
        
        land_data = []
        land_index = []
        crash_data = []
        crash_index = []
        
        land = 0
        crash = 0
        
        #split up the data
        for index, label in enumerate(labels):
            if label == 'Landed':
                land_data.append(data[index])
                land_index.append(index)
                land = land + 1
            else:
                crash_data.append(data[index])
                crash_index.append(index)
                crash = crash + 1
        
        ratio = land/crash
        
        plt.title("Spread of Data {}.".format(ratio))
        plt.scatter(land_data, land_index)
        plt.scatter(crash_data, crash_index)
        plt.xlabel('Score')
        plt.ylabel('Iteration')
        plt.show()
        
    
        