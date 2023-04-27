#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is written and completed by R.Ma.
Ver 0.1 21/02/2023
Ver 0.2 22/02/2023
Ver 0.9 23/02/2023
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

class SarsaAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        
        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")

            if np.random.rand() < epsilon:
                a = np.random.randint(0,self.n_actions)
            else:
                a = argmax(self.Q_sa[s,:])
                
        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")

            a = np.random.choice(self.n_actions, p=softmax(self.Q_sa[s,:], temp))
            
        return a
        
    def update(self,s,a,r,s_next,a_next,done):

        self.Q_sa[s,a] = self.Q_sa[s,a] + self.learning_rate * (r + self.gamma * self.Q_sa[s_next,a_next] - self.Q_sa[s,a])
        pass
        
def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = []
    budget = 0
    s_start = env.reset()
    s = s_start
    a_start = pi.select_action(s_start, policy, epsilon, temp)
    a = a_start

    while budget < n_timesteps:

        s_next, r, done = env.step(a)
        a_next = pi.select_action(s_next, policy, epsilon, temp)
        pi.update(s, a, r, s_next, a_next, done)

        if done:
            s = env.reset()
            a = pi.select_action(s, policy, epsilon, temp)
        else:
            s = s_next
            a = a_next

        rewards.append(r)
        budget += 1
        # print("Step: {}, Reward: {}".format(budget,r))
    
    if plot:
        env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=3)

    return rewards


def test():
    n_timesteps = 50000
    gamma = 1.0
    learning_rate = 0.2

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
    #print("Obtained rewards: {}".format(rewards))
    print("Average reward: {}".format(np.mean(rewards)))
    
if __name__ == '__main__':
    test()
