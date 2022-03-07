# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 00:16:58 2022

@author: Tyler
"""

import numpy as np
import matplotlib.pyplot as plt


class agent(object):
    
    def __init__(self, armedBandits,epsilon):
        self.armedBandits=armedBandits
        self.epsilon=epsilon#probability the agent will explore instead of taking an exploiting action
        self.ActionReward=np.zeros#cumulative reward each time the action was taken
        self.ActionCount=np.zeros(10)#number of times action has been taken
        self.Qt=np.zeros(10)#esimated reward of action at time step t
        self.time_step=0
        
    def exploring(self,k):
        actions=list(range(k))
        exploring_action=np.random.choice(actions)
        return exploring_action
    
    def exploiting(self,Qt):
        greedy_action=np.argmax(Qt)
        return greedy_action
        
    def action_choice(self):
        if(np.random.random()>self.epsilon and np.amax(self.Qt)!=0):#chose make greedy choices if estimated reward is not 0 and probability is higher than epsilon
            action=self.exploiting(self.Qt)
        else:
            action=self.exploring(self.armedBandits)
           
        return action
            
    def update_esimations(self,action,reward):           
        self.ActionReward[action]+=reward
        self.ActionCount[action]+=1
            
        self.Qt[action]=self.ActionReward[action]/self.ActionCount[action]
            
        self.time_step+=1
        
    def reset(self):
        self.ActionReward=np.zeros(10)
        self.ActionCount=np.zeros(10)
        self.Qt=np.zeros(10)
        self.time_step=0

class kArmedTestBed(object):
    
    def __init__(self, armedBandits,mean,sd): 
        self.armedBandits=armedBandits
        self.mean=mean
        self.sd=sd
        self.reward_distribution=np.zeros(self.armedBandits)
        self.optimal_action=0
        self.reset()
        
    def reset(self):
        #random gaussian values to use as rewards for each action
        self.reward_distribution=np.random.normal(self.mean,self.sd,self.armedBandits)
        #optimal action at that time step
        self.optimal_action=np.argmax(self.reward_distribution)
        

        
if __name__ == "__main__":
    
    kArmed=10
    steps=1000
    iterations=2000
    
    rewards=np.zeros((steps,3))
    optimal=np.zeros((steps,3))
    
    testbed=kArmedTestBed(10, 0, 1)
    agents=[agent(10,0),agent(10,0.1),agent(10,0.01)]
    agentNames=["greedy","epsilon=0.1","epsilon=0.01"]
    
    for i in range(iterations):
        testbed.reset()
        for agent in agents:
            agent.reset()
        
        for j in range(steps):
            agentCount=0
            for agent in agents:
                At=agent.action_choice()#action taken at this time step
                
                Rt=np.random.normal(testbed.reward_distribution[At],1)#reward at this timestep
                
                agent.update_esimations(At, Rt)
                
                rewards[j,agentCount]+=Rt
                if(At==testbed.optimal_action):
                    optimal[j,agentCount]+=1
                
                agentCount+=1
            
    AvgReward=rewards/iterations
    AvgOptimal=optimal/iterations
    
    plt.title("10-Armed TestBed - Average Rewards")
    plt.plot(AvgReward)
    plt.ylabel('Average Reward')
    plt.xlabel('Steps')
    plt.legend(agentNames, loc=4)
    plt.show()


    plt.title("10-Armed TestBed - % Optimal Action")
    plt.plot(AvgOptimal * 100)
    plt.ylim(0, 100)
    plt.ylabel('% Optimal Action')
    plt.xlabel('Steps')
    plt.legend(agentNames, loc=4)
    plt.show()
        
        
    
    