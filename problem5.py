# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 20:19:33 2022

@author: Tyler
"""

import numpy as np
import matplotlib.pyplot as plt

class agent(object):
    def __init__(self):
        #all 48 states has the value for each action initialised at 0
        self.q_table=np.zeros((48,4))
                

    def update_q_table(self,q_table,state,action,reward,next_state_value,gamma=1,alpha=0.1):
        updated_value=q_table[state,action]+alpha*(reward+(gamma*next_state_value)-q_table[state,action])
        q_table[state,action]=updated_value
        return q_table
    
    def sarsa(self,gamma=1,alpha=0.1):
        reward=0
        return reward
    
    def q_learning(self,episodes=500,gamma=1,alpha=0.1,epsilon=0.1):
        for episode in range(episodes):
            state=0
            for episode in range(episodes):
                env=environment()
                terminating_state=False
                cum_reward=0
                while(terminating_state==False):
                    state=env.update_agent_position(-1)
                    action=self.greedy_policy(state)
                    next_state=env.update_agent_position(action)
                    reward,terminating_state=env.get_reward()
                    cum_reward+=reward
        
        return action
    
    def greedy_policy(self,state,epsilon=0.1):
        if(np.random.random()<epsilon):
            action=np.random.choice(4) #up=0 left=1 right=2 down=3
        else:
            action=np.argmax(self.q_table[state,:])
        return action
    
class cliffwalkingenvironment(object):
    def __init__(self):
        self.state_space=self.build_environment()
        self.agentx=0
        self.agenty=3
        
    def build_environment(self):
        environment=-np.ones((4,12))
        environment[3,1:11]=-100
        return environment
    
    def update_agent_position(self,action):
        if(action==0 and self.agenty>0):#up
            self.agentx=self.agenty-1
        elif(action==1 and self.agentx>0):#left
            self.agentx=self.agentx-1
        elif(action==2 and self.agentx<11):#right
            self.agentx=self.agentx+1
        elif(action==3 and self.agenty<3):#down
            self.agentx=self.agenty+1
        state=12*self.agentx+self.agenty
        return state
        
    def get_reward(self):
        #gets reward for the transition. terminates the episode at the goal state or cliff
        terminating_state=False
        reward=-1
        if(self.state_space[self.agenty,self.agentx]==-100):
            terminating_state=True
            reward=-100
        elif(self.agenty==3 and self.agentx==11):
            terminating_state=True
            
        return reward,terminating_state
        
        
        
if __name__ == "__main__":
    environment=cliffwalkingenvironment()
    sarsa=agent()
    q_learning=agent()