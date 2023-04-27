#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is written and completed by R.Ma.
Ver 0.1 22/02/2023
Ver 0.2 26/02/2023
Ver 0.3 27/02/2023
Ver 0.4 01/03/2023
Ver 0.5 02/03/2023
Ver 0.9 03/03/2023

"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax


class MonteCarloAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
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

        T_ep = len(rewards)
        G = np.zeros(T_ep + 1)
        G[T_ep] = 0

        for i in reversed(range(T_ep)):
            G[i] = rewards[i] + self.gamma * G[i + 1]
            self.Q_sa[states[i], actions[i]] += self.learning_rate * (G[i] - self.Q_sa[states[i], actions[i]])


def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma,
                policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep '''

    env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = []
    budget = 0  # timestep counter
    while budget < n_timesteps:

        s = env.reset()
        states = [s]
        actions = []
        rewards_episode = []

        for t in range(max_episode_length):

            a_t = pi.select_action(s, policy, epsilon, temp)
            s_next, r_t, done = env.step(a_t)
            actions.append(a_t)
            states.append(s_next)
            rewards_episode.append(r_t)
            s = s_next
            budget += 1
            if done or budget >= n_timesteps:
                break
            #print("Timestep: {}, reward: {}".format(budget, r_t))

        pi.update(states, actions, rewards_episode, done)

        rewards.extend(rewards_episode)

    if plot:
        env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True, step_pause=3)

    return rewards



def test():
    n_timesteps = 50000
    max_episode_length = 150
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = True

    rewards = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma,
                   policy, epsilon, temp, plot)
    print("Obtained rewards: {}".format(rewards))
    print(len(rewards))
    print("Average reward: {}".format(np.mean(rewards)))

if __name__ == '__main__':
    test()
