# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 23:21:11 2018

@author: Biswadeep Sen
"""

import numpy as np

import matplotlib.pyplot as plt

VALUES = [0]*7


class state:
    def __init__(self,name,nbhrs = []):
        self.name = name
        self.nbhrs = nbhrs
        self.value = 0
        
    def choose_nbhr(self):
        if self.name == 0 or self.name == 6:
            return None
        nbhrs_list =  [i[0] for i in self.nbhrs]
        probs = [i[1] for i in self.nbhrs]
        chosen_nbhr =  np.random.choice(nbhrs_list,p = probs)
        return chosen_nbhr
    
s0 = state(0)
s1 = state(1)
s2 = state(2)
s3 = state(3)
s4 = state(4)
s5 = state(5)
s6 = state(6)


s1.nbhrs = [(s0,0.6),(s2,0.2),(s3,0.2)]
s2.nbhrs = [(s4,0.5),(s1,0.3),(s5,0.2)]
s3.nbhrs = [(s1,0.2),(s4,0.3),(s5,0.5)]
s4.nbhrs = [(s2,0.6),(s6,0.4)]
s5.nbhrs = [(s6,0.6),(s3,0.4)]


def TD(values,start,alpha,gamma = 1):
    state = start
    trajectory = [start]
    while True:
        next_state = state.choose_nbhr()#perform the action and get the next state
        if next_state.name == 6:#TD update values
            values[state.name] += alpha * (1 + gamma*values[next_state.name] - values[state.name])
        else:
            values[state.name] += alpha * (0 + gamma*values[next_state.name] - values[state.name])
        state = next_state
        trajectory.append(state)
        if state.name == 0 or state.name == 6:#If state is terminal break
            break
    return trajectory



def Monte_Carlo(values,start,alpha,gamma=1):
    #We generate an episode
    state = start
    rewards = []
    trajectory = [state]
    while True:#Execute a run until you see a fina; state
        state = state.choose_nbhr()
        trajectory.append(state)
        if state.name == 6:
            rewards.append(1)
            break
        elif state.name == 0:
            rewards.append(0)
            break
        else:
            rewards.append(0)
    G = rewards[-1]
    
    for state in reversed(trajectory[:-1]):#For each value of the run we update the values according to the MC rule
        
        values[state.name] = values[state.name] + alpha * (G - values[state.name])
        G = gamma*G
    
    return trajectory
        

def update(method,episodes,a,b):
    VALUES = [0]*7
    if method == "TD":
        for i in range(episodes):
            TD(VALUES,s1,a,b)
    else:
        for i in range(episodes):
            Monte_Carlo(VALUES,s1,a,b)
    return VALUES

#episode_list = list(range(0,1001,100))
#alpha_list = [0.01,0.05,0.1,0.15]
#
#
#colors = ["red","green","blue","yellow"]
#
#for j in range(len(alpha_list)):
#    values = []
#    for i in episode_list:
#        k = update("MC",i,alpha_list[j],0.5)
#        values.append(k[5])
#    plt.plot(episode_list,values,colors[j],label = "alpha = " + str(alpha_list[j]))
#    plt.xlabel("episodes")
#    plt.ylabel("state 5 value")
#    plt.legend()
#plt.title("state 5 with gamma = 0.5")



    



        
    
    



