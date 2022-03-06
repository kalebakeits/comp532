import numpy as np
import matplotlib.pyplot as plt

class agent(object):
    def __init__(self):
        #all 48 states has the value for each action initialised at 0
        self.q_table=np.zeros((48,4))
        self.states=[]
        self.episode_reward=np.zeros(500)
                

    def update_q_table(self,q_table,state,action,reward,next_state_value,gamma=1,alpha=0.1):
        updated_value=q_table[state,action]+alpha*(reward+(gamma*next_state_value)-q_table[state,action])
        q_table[state,action]=updated_value
        return q_table
    
    def sarsa(self,episodes=500,gamma=1,alpha=0.1,epsilon=0.1):
        cliff=cliffwalkingenvironment()#build environment
        terminating_state=False
        episode_rewards=np.zeros(episodes)
        for episode in range(episodes):
            cum_reward=0
            cliff.reset()
            terminating_state=False
            curr_state=cliff.get_state()
            action=self.action_probs(curr_state)
            print("action taken is ",action)
            while(terminating_state==False):
                cliff.update_agent_position(action)
                new_state=cliff.get_state()
                print("current state is ",curr_state,"new state is ",new_state)
                reward,terminating_state=cliff.get_reward()
                print("reward is ", reward)
                self.q_table[curr_state,action]+=alpha*(reward+gamma*np.amax(self.q_table[new_state,:])-self.q_table[curr_state,action])
                print("episode ",episode)
                if(terminating_state==True and reward==-100):
                    cum_reward=-100
                else:
                    cum_reward+=reward#reward so far for the episode
            episode_rewards[episode]=cum_reward
                
        return episode_rewards
    
    def q_learning(self,episodes=500,gamma=1,alpha=0.1,epsilon=0.1):
        cliff=cliffwalkingenvironment()#build environment
        terminating_state=False
        episode_rewards=np.zeros(episodes)
        for episode in range(episodes):
            cum_reward=0
            cliff.reset()
            terminating_state=False
            while(terminating_state==False):
                curr_state=cliff.get_state()
                action=self.greedy_policy(curr_state)
                print("action taken is ",action)
                cliff.update_agent_position(action)
                new_state=cliff.get_state()
                print("current state is ",curr_state,"new state is ",new_state)
                reward,terminating_state=cliff.get_reward()
                print("reward is ", reward)                   
                self.q_table[curr_state,action]+=alpha*(reward+gamma*np.amax(self.q_table[new_state,:])-self.q_table[curr_state,action])
                print("episode ",episode)
                if(terminating_state==True and reward==-100):
                    cum_reward=-100
                else:
                    cum_reward+=reward#reward so far for the episode
            episode_rewards[episode]=cum_reward
                
        return episode_rewards
    
    def greedy_policy(self,state,epsilon=0.1):
        if(np.random.random()<epsilon):
            action=np.random.choice(4) #up=0 left=1 right=2 down=3
        else:
            action=np.argmax(self.q_table[state,:])
        return action
    
    def action_probs(self, state, epsilon=0.1):
        next_state_probs = [epsilon/4] * 4
        best_action = self.greedy_policy(state)
        next_state_probs[best_action] += (1.0 - epsilon)
        if np.random.random()>epsilon:
            return best_action
        next_state_probs.remove(best_action)
        return next_state_probs[np.random.choice(3)]
    
class cliffwalkingenvironment(object):
    def __init__(self):
        self.state_space=self.build_environment()
        self.agentx=0
        self.agenty=3
        
    def get_state(self):
        state=12*self.agenty+self.agentx
        return state        
    
    def build_environment(self):
        environment=-np.ones((4,12))
        environment[3,1:11]=-100
        return environment
    
    def update_agent_position(self,action):
        if(action==0 and self.agenty>0):#up
            self.agenty=self.agenty-1
        elif(action==1 and self.agentx>0):#left
            self.agentx=self.agentx-1
        elif(action==2 and self.agentx<11):#right
            self.agentx=self.agentx+1
        elif(action==3 and self.agenty<3):#down
            self.agenty=self.agenty+1
        
    def get_reward(self):
        #gets reward for the transition. terminates the episode at the goal state or cliff
        terminating_state=False
        reward=-1
        if(self.state_space[self.agenty,self.agentx]==-100):
            terminating_state=True

        elif(self.agenty==3 and self.agentx==11):
            terminating_state=True
            
        return self.state_space[self.agenty,self.agentx],terminating_state
    
    def reset(self):
        self.agentx=0
        self.agenty=3
        
        
        
if __name__ == "__main__":
    environment=cliffwalkingenvironment()
    q_learning=agent()
    sarsa_agent=agent()
    
    q_episode_rewards=q_learning.q_learning()
    sarsa_episode_rewards=sarsa_agent.sarsa()
    avg_q_reward=np.zeros((500,1))
    avg_sarsa_reward=np.zeros((500,1))
    for i in range(10,500):
        avg=sum(q_episode_rewards[i-10:i])/10
        avg_q_reward[i,0]=avg
    
    for i in range(10, 500):
        avg=sum(sarsa_episode_rewards[i-10:i])/10
        avg_sarsa_reward[i,0]=avg
        
    plt.title("Sarsa vs Q Learning - Average Rewards")
    plt.plot(avg_q_reward[10:500])
    plt.plot(avg_sarsa_reward[10:500])
    plt.ylabel('Sum of rewards during episode')
    plt.xlabel('Episodes')
    #plt.legend(agentNames, loc=4)
    plt.show()
