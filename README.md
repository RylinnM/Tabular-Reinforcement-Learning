# Tabular-Reinforcement-Learning
A collection of several tabular reinforcement learning methods that are implemented from scratch.

This project contains several implementations of tabular reinforcement learning algorithms, including dynamic programming, Q-learning, SARSA, n-step Q-learning, and Monte Carlo. These algorithms are implemented in Python and can be used to train agents to solve various reinforcement learning problems.

## Installation

To use this project, you should have Python 3.x installed on your system. You can then clone the repository to your local machine using the following command:

git clone https://github.com/RylinnM/Tabular-Reinforcement-Learning.git


## Usage

The main entry point for the project is the `experiment.py` script. You can use this script to train agents using the different algorithms implemented in the project. Multiple options are available in the code to enable various configurations.

The available options are:

- `--algorithm`: The reinforcement learning algorithm to use (dp, qlearning, sarsa, nstepq, or mc).

- `--num_episodes`: The number of episodes to train the agent for.
- `--gamma`: The discount factor to use in the reinforcement learning algorithm.
- `--alpha`: The learning rate to use in the reinforcement learning algorithm.
- `--n`: The number of steps to use in n-step Q-learning.
- `--render`: Whether to render the environment during training.








