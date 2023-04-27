#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Written and commented by: R.Ma
Ver 0.1 20/02/2023
Ver 0.2 22/02/2023
Ver 0.9 24/02/2023
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax
from Helper import linear_anneal

class QLearningAgent:

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
        
    def update(self,s,a,r,s_next,done):

        self.Q_sa[s,a] = self.Q_sa[s,a] + self.learning_rate * (r + self.gamma * np.max(self.Q_sa[s_next,:]) - self.Q_sa[s,a])
        pass

def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep '''

    rewards = []

    env = StochasticWindyGridworld(initialize_model=False)

    agent = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    agent.Q_sa = np.zeros((env.n_states, env.n_actions))

    t = 0
    s_start = env.reset()
    s = s_start
    while t < n_timesteps:

        done = False
        a = agent.select_action(s, policy, epsilon, temp)

        s_next, r, done = env.step(a)
        agent.update(s, a, r, s_next, done)

        if done:
            s = env.reset()
        else:
            s = s_next
        rewards.append(r)
        # print("timestep: {}, reward: {}".format(t, rewards[t]))
        t += 1
    if plot:
        env.render(Q_sa=agent.Q_sa, plot_optimal_policy=True, step_pause=3)

    return rewards

# The following function is used to test the effect of parameter annealing. To run this, follow the instructions in the experiment.py


def q_learning_with_annealing(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, percent=0.5):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep '''

    rewards = []

    env = StochasticWindyGridworld(initialize_model=False)

    agent = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    agent.Q_sa = np.zeros((env.n_states, env.n_actions))

    t = 0
    s_start = env.reset()
    s = s_start
    while t < n_timesteps:

        done = False
        epsilon = linear_anneal(t, n_timesteps, 1.0, 0.02, percent)
        # print(epsilon)
        temp = linear_anneal(t, n_timesteps, 1.0, 0.01, percent)
        a = agent.select_action(s, policy, epsilon, temp)

        s_next, r, done = env.step(a)
        agent.update(s, a, r, s_next, done)

        if done:
            s = env.reset()
        else:
            s = s_next
        rewards.append(r)
        # print("timestep: {}, reward: {}".format(t, rewards[t]))
        t += 1
    if plot:
        env.render(Q_sa=agent.Q_sa, plot_optimal_policy=True, step_pause=3)

    return rewards


def test():

    # show the rewards in different time steps

    n_timesteps = 50000
    gamma = 1.0
    learning_rate = 0.01 #origianl 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
    #print("Obtained rewards: {}".format(rewards))
    print("Average reward: {}".format(np.mean(rewards)))
    print(len(rewards))

if __name__ == '__main__':
    test()
