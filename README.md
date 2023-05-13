# Tabular-Reinforcement-Learning
A collection of several tabular reinforcement learning methods implemented from scratch.

This project contains several implementations of tabular reinforcement learning algorithms, including dynamic programming, Q-learning, SARSA, n-step Q-learning, and Monte Carlo. These algorithms are implemented in Python and can be used to train agents to solve various reinforcement learning problems.

## Installation

To use this project, you should have Python 3.x installed on your system. You can then clone the repository to your local machine using the following command:

git clone https://github.com/RylinnM/Tabular-Reinforcement-Learning.git


## Usage

The main entry point for the project is the `experiment.py` script. You can use this script to train agents using the different algorithms implemented in the project. Multiple options are available in the code to enable various configurations. Please be advised that these options have to be configured in the script itself instead of argument in the command line.

The available options are:

- `--backup`: The reinforcement learning algorithm backup to use (dp, qlearning, sarsa, nstepq, or mc).
- `--n_repetitions`: The number of repetitions to train the algorithm for.
- `--n_timesteps`: The number of timesteps to train the agent for.
- `--gamma`: The discount factor to use in the reinforcement learning algorithm.
- `--learning_rate`: The learning rate to use in the reinforcement learning algorithm.
- `--n`: The number of steps to use in n-step Q-learning.
- `--max_episode_length`: The maximum episode length that can be reached during training.
- `--policy`: The action selection policy used.
- `--epsilon`: Parameter for the epsilon-greedy policy.
- `--temp`: Parameter for the softmax policy.
- `--smoothing_window`: The window selection used to smooth the graph.
- `--plot`: Whether or not plot the figure. (T/F).
- `--percent`: The percent of annealing of selection policy parameters during training.








