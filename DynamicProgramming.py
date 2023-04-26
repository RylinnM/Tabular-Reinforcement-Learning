#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Thomas Moerland

This file is written and completed by R.Ma.
Ver 0.1 18/02/2023
Ver 0.2 22/02/2023
Ver 0.9 01/03/2023
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Environment_62 import StochasticWindyGridworld_62
from Helper import argmax


class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states, n_actions))

    def select_action(self, s):
        ''' Returns the greedy best action in state s '''
        a = argmax(self.Q_sa[s, :])
        return a

    def update(self, s, a, p_sas, r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        self.Q_sa[s, a] = np.sum(p_sas * (r_sas + self.gamma * np.max(self.Q_sa, axis=1)))


def Q_value_iteration(env, gamma=1.0, threshold=0.01):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''

    agent = QValueIterationAgent(env.n_states, env.n_actions, gamma)

    max_error = np.inf
    i = 0
    while max_error > threshold:
        Q_sa_previous = np.copy(agent.Q_sa)
        for s in range(env.n_states):
            for a in range(env.n_actions):
                p_sas, r_sas = env.model(s, a)
                agent.update(s, a, p_sas, r_sas)
                max_error_new = np.max(np.abs(agent.Q_sa - Q_sa_previous))
                if max_error_new < max_error:
                    max_error = max_error_new
                else:
                    max_error = max_error
                i += 1
                # print("Q-value iteration, iteration {}, max error {}".format(i, max_error))

    return agent, i, agent.Q_sa


def experiment():
    iterations = 0
    gamma = 1.0
    threshold = 0.01
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent, iterations, Q_table= Q_value_iteration(env, gamma, threshold)
    repetitions = 1
    average_reward = 0

    for i in range(repetitions):
        steps = 0
        done = False
        s = env.reset()
        rewards = 0
        while not done:
            a = QIagent.select_action(s)
            s_next, r, done = env.step(a)
            steps += 1
            rewards += r
            s = s_next
            #print("steps: {}, rewards: {}".format(steps, rewards))
            print("steps: {}, rewards: {}".format(steps, rewards))
            env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.5)

            mean_reward_per_timestep = rewards / steps
        average_reward += mean_reward_per_timestep
    average_reward = average_reward / repetitions
    #print("Mean reward per timestep: {}".format(mean_reward_per_timestep))
    print("Average reward per timestep: {}".format(average_reward))

    # The following code is for the altered environment with the goal at (6,2), uncomment to run
    # Note that the StochasticWindyGridworld_62 class is in the Environment_62.py file, which is not uploaded
    # to the submission. You can copy an additional environment.py file and manually change the position to (6,2).
    """
    env_62 = StochasticWindyGridworld_62(initialize_model=True)
    env_62.render()
    repetitions = 1
    average_reward_62 = 0
    # calculate the steps needed in the optimal policy
    for i in range(repetitions):
        steps = 0
        done = False
        s = env_62.reset()
        rewards = 0
        while not done:
            a = QIagent.select_action(s)
            s_next, r, done = env_62.step(a)
            steps += 1
            rewards += r
            s = s_next
            print("steps: {}, rewards: {}".format(steps, rewards))
            env_62.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.5)
"""

if __name__ == '__main__':
    experiment()

