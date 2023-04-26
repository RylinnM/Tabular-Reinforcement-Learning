#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland

This file is written and completed by R.Ma.
Ver 0.1 20/02/2023
Ver 0.2 22/02/2023
Ver 0.3 27/02/2023
Ver 0.4 01/03/2023
Ver 0.5 02/03/2023
Ver 0.9 03/03/2023
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax


class NstepQLearningAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, n):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n = n
        self.Q_sa = np.zeros((n_states, n_actions))

    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):

        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")

            if np.random.rand() < epsilon:
                a = np.random.randint(0, self.n_actions)
            else:
                a = argmax(self.Q_sa[s, :])

        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")
            a = np.random.choice(self.n_actions, p=softmax(self.Q_sa[s, :], temp))

        return a

    def update(self, states, actions, rewards, done):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''

        T_ep = len(states) - 1
        G = np.zeros(T_ep + 1)

        for t in range(T_ep):
            m = min(T_ep - t, self.n)

            if t + m == T_ep:
                G[t] = sum([self.gamma ** i * rewards[t + i] for i in range(m)])
            else:
                G[t] = sum([self.gamma ** i * rewards[t + i] for i in range(m)]) + (self.gamma ** m) * np.max(self.Q_sa[states[t + m]])

            self.Q_sa[states[t], actions[t]] += self.learning_rate * (G[t] - self.Q_sa[states[t], actions[t]])


def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma,
             policy='egreedy', epsilon=None, temp=None, plot=True, n=3):
    ''' runs a single repetition of an MC rl agentnh
    Return: rewards, a vector with the observed rewards at each timestep '''

    env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma, n)
    rewards = []
    b = 0
    while b < n_timesteps:
        s = env.reset()
        states = [s]
        actions = []
        rewards_ep = []
        for t in range(max_episode_length):

            a_t = pi.select_action(s, policy, epsilon, temp)
            s_next, r_t, done = env.step(a_t)
            states.append(s_next)
            actions.append(a_t)
            rewards_ep.append(r_t)
            rewards.append(r_t)
            s = s_next
            b += 1
            if done or b >= n_timesteps:
                break
            #print("time step: {}, reward: {}".format(b, r_t))

        pi.update(states, actions, rewards_ep, done)
    if plot:
        env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True, step_pause=3)


    return rewards


def test():
    n_timesteps = 50000
    max_episode_length = 150
    gamma = 1.0
    learning_rate = 0.25
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
    print("Obtained rewards: {}".format(rewards))
    print(len(rewards))
    print("Average reward: {}".format(np.mean(rewards)))
    
if __name__ == '__main__':
    test()
