#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Written and modified by R.Ma.
"""

import numpy as np
import time

from Q_learning import q_learning, q_learning_with_annealing
from SARSA import sarsa
from MonteCarlo import monte_carlo
from Nstep import n_step_Q
from Helper import LearningCurvePlot, smooth

def average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, gamma, policy='egreedy', 
                    epsilon=None, temp=None, smoothing_window=51, plot=False, n=5, percent=None):

    reward_results = np.empty([n_repetitions,n_timesteps]) # Result array
    now = time.time()
    
    for rep in range(n_repetitions): # Loop over repetitions
        if backup == 'q':
            rewards = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
        elif backup == 'sarsa':
            rewards = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
        elif backup == 'mc':
            rewards = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
        elif backup == 'nstep':
            rewards = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
        elif backup == 'an':
            rewards = q_learning_with_annealing(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, percent)

        reward_results[rep] = rewards

    print('Running one setting takes {} minutes'.format((time.time()-now)/60))
    learning_curve = np.mean(reward_results,axis=0) # average over repetitions
    learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return learning_curve  

def experiment():
    ####### Settings
    # Experiment    
    n_repetitions = 50 #original 10
    smoothing_window = 1001
    plot = False # Plotting is very slow, switch it off when we run repetitions #original false
    
    # MDP    
    n_timesteps = 50000
    max_episode_length = 150
    gamma = 1.0

    # Parameters we will vary in the experiments, set them to some initial values: 
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.05
    temp = 1.0
    # Back-up & update
    backup = 'q' # 'q' or 'sarsa' or 'mc' or 'nstep'
    learning_rate = 0.25
    # n = 5 # only used when backup = 'nstep'
        
    # Nice labels for plotting
    backup_labels = {'q': 'Q-learning',
                  'sarsa': 'SARSA',
                  'mc': 'Monte Carlo',
                  'nstep': 'n-step Q-learning'}
    
    ####### Experiments
    
    #### Assignment 1: Dynamic Programming
    # Execute this assignment in DynamicProgramming.py
    optimal_average_reward_per_timestep = 1.31 # set the optimal average reward per timestep you found in the DP assignment here
    
    #### Assignment 2: Effect of exploration
    policy = 'egreedy'
    epsilons = [0.02,0.1,0.3]
    learning_rate = 0.25
    percents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    backup = 'q'
    Plot = LearningCurvePlot(title = 'Exploration: $\epsilon$-greedy versus softmax exploration')    
    for epsilon in epsilons:        
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                              gamma, policy, epsilon, temp, smoothing_window, plot, n=5)
        Plot.add_curve(learning_curve,label=r'$\epsilon$-greedy, $\epsilon $ = {}'.format(epsilon))    
    policy = 'softmax'
    temps = [0.01,0.1,1.0]
    for temp in temps:
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                              gamma, policy, epsilon, temp, smoothing_window, plot, n=5)
        Plot.add_curve(learning_curve,label=r'softmax, $ \tau $ = {}'.format(temp))
    Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")
    Plot.save('exploration.png')
    # The following code is for the bonus annealing exploration
    """
    backup = 'an'
    epsilon = 0.02
    temp = 0.01
    policy = 'egreedy'
    Plot = LearningCurvePlot(title = 'Annealing: $\epsilon$-greedy exploration w/o parameter annealing')
    learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate,
                                              gamma, policy, epsilon, temp, smoothing_window, plot, n=5)
    Plot.add_curve(learning_curve, label=r'$\epsilon$-greedy, $\epsilon $ = {}'.format(epsilon))
    for percent in percents:
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate,
                                              gamma, policy, epsilon, temp, smoothing_window, plot, n=5, percent=percent)
        Plot.add_curve(learning_curve,label=r'$\epsilon$-greedy with annealing p.t. = {}'.format(percent))
    Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")
    Plot.save('epsilon-greedy_annealing.png')

    policy = 'softmax'
    Plot = LearningCurvePlot(title='Annealing: softmax exploration w/o parameter annealing')
    learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate,
                                              gamma, policy, epsilon, temp, smoothing_window, plot, n=5)
    Plot.add_curve(learning_curve, label=r'softmax, $\tau $ = {}'.format(temp))
    for percent in percents:
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate,
                                                  gamma, policy, epsilon, temp, smoothing_window, plot, n=5,
                                                  percent=percent)
        Plot.add_curve(learning_curve, label=r'softmax with annealing p.t. = {}'.format(percent))
    Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")
    Plot.save('softmax_annealing.png')
    """
    ###### Assignment 3: Q-learning versus SARSA
    policy = 'egreedy'
    epsilon = 0.1 # set epsilon back to original value 
    learning_rates = [0.02,0.1,0.4]
    backups = ['q','sarsa']
    Plot = LearningCurvePlot(title = 'Back-up: on-policy versus off-policy')    
    for backup in backups:
        for learning_rate in learning_rates:
            learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                                  gamma, policy, epsilon, temp, smoothing_window, plot)
            Plot.add_curve(learning_curve,label=r'{}, $\alpha$ = {} '.format(backup_labels[backup],learning_rate))
    Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")
    Plot.save('on_off_policy.png')
    
    # ##### Assignment 4: Back-up depth
    policy = 'egreedy'
    epsilon = 0.1 # set epsilon back to original value
    learning_rate = 0.25
    backup = 'nstep'
    ns = [1,3,10,30]
    Plot = LearningCurvePlot(title = 'Back-up: depth')    
    for n in ns:
        learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                              gamma, policy, epsilon, temp, smoothing_window, plot, n)
        Plot.add_curve(learning_curve,label=r'{}-step Q-learning'.format(n))
    backup = 'mc'
    learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                          gamma, policy, epsilon, temp, smoothing_window, plot, n)
    Plot.add_curve(learning_curve,label='Monte Carlo')        
    Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")
    Plot.save('depth.png')

if __name__ == '__main__':
    experiment()
