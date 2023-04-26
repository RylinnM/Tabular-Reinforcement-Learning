#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""
import matplotlib
matplotlib.use('Qt5Agg') # 'TkAgg'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle,Arrow

class StochasticWindyGridworld:
    ''' Stochastic version of WindyGridworld 
        (based on Sutton & Barto, Example 6.5 at page 130, see http://incompleteideas.net/book/RLbook2020.pdf)
        Compared to the book version, the vertical wind is now stochastic, and only blows 80% of times
    '''
    
    def __init__(self,initialize_model=True):
        self.height = 7 # 7
        self.width = 10 # 10
        self.shape = (self.width, self.height)
        self.n_states = self.height * self.width
        self.n_actions = 4
        self.action_effects = {
                0: (0, 1),  # up
                1: (1, 0),   # right
                2: (0, -1),   # down
                3: (-1, 0),  # left
                }
        self.start_location = (0,3)
        self.winds = (0,0,0,1,1,1,2,2,1,0)
        self.wind_blows_proportion = 0.8         

        self.reward_per_step = -1.0 # default reward on every step that does not reach a goal
        self.goal_locations = [[7,3]] # [[6,2]] a vector specifying the goal locations in [[x1,y1],[x2,y2]] format
        self.goal_rewards = [40] # a vector specifying the associated rewards with the goals in self.goal_locations, in [r1,r2] format
        
        # Initialize model
        self.initialize_model = initialize_model
        if self.initialize_model:
            self._construct_model()
            
        # Initialize figures
        self.fig = None
        self.Q_labels = None
        self.arrows = None
        
        # Set agent to the start location
        self.reset() 

    def reset(self):
        ''' set the agent back to the start location '''
        self.agent_location = np.array(self.start_location)
        s = self._location_to_state(self.agent_location)
        return s
    
    def step(self,a):
        ''' Forward the environment based on action a, really affecting the agent location  
        Returns the next state, the obtained reward, and a boolean whether the environment terminated '''
        self.agent_location += self.action_effects[a] # effect of action
        self.agent_location = np.clip(self.agent_location,(0,0),np.array(self.shape)-1) # bound within grid
        if np.random.rand() < self.wind_blows_proportion: # apply effect of wind
            self.agent_location[1] += self.winds[self.agent_location[0]] # effect of wind
        self.agent_location = np.clip(self.agent_location,(0,0),np.array(self.shape)-1) # bound within grid
        s_next = self._location_to_state(self.agent_location)    
        
        # Check reward and termination
        goal_present = np.any([np.all(goal_location == self.agent_location) for goal_location in self.goal_locations])
        if goal_present:
            goal_index = np.where([np.all(goal_location == self.agent_location) for goal_location in self.goal_locations])[0][0]
            done = True
            r = self.goal_rewards[goal_index]
        else: 
            done = False
            r = self.reward_per_step           
            
        return s_next, r, done  

    def model(self,s,a):
        ''' Returns vectors p(s'|s,a) and r(s,a,s') for given s and a.
        Only simulates, does not affect the current agent location '''
        if self.initialize_model:
            return self.p_sas[s,a], self.r_sas[s,a]
        else:
            raise ValueError("set initialize_model=True when creating Environment")
            

    def render(self,Q_sa=None,plot_optimal_policy=False,step_pause=0.001):
        ''' Plot the environment 
        if Q_sa is provided, it will also plot the Q(s,a) values for each action in each state
        if plot_optimal_policy=True, it will additionally add an arrow in each state to indicate the greedy action '''
        # Initialize figure
        if self.fig == None:
            self._initialize_plot()
            
        # Add Q-values to plot
        if Q_sa is not None:
            # Initialize labels
            if self.Q_labels is None:
                self._initialize_Q_labels()
            # Set correct values of labels
            for state in range(self.n_states):
                for action in range(self.n_actions):
                    self.Q_labels[state][action].set_text(np.round(Q_sa[state,action],1))

        # Add arrows of optimal policy
        if plot_optimal_policy and Q_sa is not None:
            self._plot_arrows(Q_sa)
            
        # Update agent location
        self.agent_circle.center = self.agent_location+0.5
            
        # Draw figure
        plt.pause(step_pause)    

    def _state_to_location(self,state):
        ''' bring a state index to an (x,y) location of the agent '''
        return np.array(np.unravel_index(state,self.shape))
    
    def _location_to_state(self,location):
        ''' bring an (x,y) location of the agent to a state index '''
        return np.ravel_multi_index(location,self.shape)
        
    def _construct_model(self):
        ''' Constructs full p(s'|s,a) and r(s,a,s') arrays
            Stores these in self.p_sas and self.r_sas '''
            
        # Initialize transition and reward functions
        p_sas = np.zeros((self.n_states,self.n_actions,self.n_states))
        r_sas = np.zeros((self.n_states,self.n_actions,self.n_states)) + self.reward_per_step # set all rewards to the default value
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                s_location = self._state_to_location(s)  
                    
                # if s is goal state (terminal) make it a self-loop without rewards
                state_is_a_goal = np.any([np.all(goal_location == s_location) for goal_location in self.goal_locations])
                if state_is_a_goal: 
                    # Make actions from this state a self-loop with 0 reward.
                    p_sas[s,a,s] = 1.0 
                    r_sas[s,a,] = np.zeros(self.n_states)  
                else:
                    # check what happens if the wind blows:
                    next_location_with_wind = np.copy(s_location) 
                    next_location_with_wind += self.action_effects[a] # effect of action
                    next_location_with_wind = np.clip(next_location_with_wind,(0,0),np.array(self.shape)-1) # bound within grid
                    next_location_with_wind[1] += self.winds[next_location_with_wind[0]] # Apply effect of wind
                    next_location_with_wind = np.clip(next_location_with_wind,(0,0),np.array(self.shape)-1) # bound within grid
                    next_state_with_wind = self._location_to_state(next_location_with_wind)   
                    
                    # Update p_sas and r_sas
                    p_sas[s,a,next_state_with_wind] += self.wind_blows_proportion
                    for (i,goal) in enumerate(self.goal_locations):
                        if np.all(next_location_with_wind == goal): # reached a goal!
                            r_sas[s,a,next_state_with_wind]  = self.goal_rewards[i]
                    
                    # check what happens if the wind does not blow:
                    next_location_without_wind = np.copy(s_location)
                    next_location_without_wind += self.action_effects[a] # effect of action
                    next_location_without_wind = np.clip(next_location_without_wind,(0,0),np.array(self.shape)-1) # bound within grid
                    next_state_without_wind = self._location_to_state(next_location_without_wind)
    
                    # Update p_sas and r_sas
                    p_sas[s,a,next_state_without_wind] += (1-self.wind_blows_proportion)
                    for (i,goal) in enumerate(self.goal_locations):
                        if np.all(next_state_without_wind == goal): # reached a goal!
                            r_sas[s,a,next_state_without_wind]  = self.goal_rewards[i] 

        self.p_sas = p_sas
        self.r_sas = r_sas
        return 

    def _initialize_plot(self):
        self.fig,self.ax = plt.subplots()#figsize=(self.width, self.height+1)) # Start a new figure
        self.ax.set_xlim([0,self.width])
        self.ax.set_ylim([0,self.height]) 
        self.ax.axes.xaxis.set_visible(False)
        self.ax.axes.yaxis.set_visible(False)

        for x in range(self.width):
            for y in range(self.height):
                self.ax.add_patch(Rectangle((x, y),1,1, linewidth=0, facecolor='k',alpha=self.winds[x]/4))       
                self.ax.add_patch(Rectangle((x, y),1,1, linewidth=0.5, edgecolor='k', fill=False))     

        self.ax.axvline(0,0,self.height,linewidth=5,c='k')
        self.ax.axvline(self.width,0,self.height,linewidth=5,c='k')
        self.ax.axhline(0,0,self.width,linewidth=5,c='k')
        self.ax.axhline(self.height,0,self.width,linewidth=5,c='k')

        # Indicate start state
        self.ax.add_patch(Rectangle(self.start_location,1.0,1,0, linewidth=0, facecolor='b',alpha=0.2))
        self.ax.text(self.start_location[0]+0.05,self.start_location[1]+0.75, 'S', fontsize=20, c='b')

        # Indicate goal states
        for i in range(len(self.goal_locations)): 
            if self.goal_rewards[i] >= 0:
                colour = 'g'
                text = '+{}'.format(self.goal_rewards[i])
            else:
                colour = 'r'
                text = '{}'.format(self.goal_rewards[i])
            self.ax.add_patch(Rectangle(self.goal_locations[i],1.0,1,0, linewidth=0, facecolor=colour,alpha=0.2))
            self.ax.text(self.goal_locations[i][0]+0.05,self.goal_locations[i][1]+0.75,text, fontsize=20, c=colour)

        # Add agent
        self.agent_circle = Circle(self.agent_location+0.5,0.3)
        self.ax.add_patch(self.agent_circle)
        
    def _initialize_Q_labels(self):
        self.Q_labels = []
        for state in range(self.n_states):
            state_location = self._state_to_location(state)
            self.Q_labels.append([])
            for action in range(self.n_actions):
                plot_location = np.array(state_location) + 0.42 + 0.35 * np.array(self.action_effects[action])
                next_label = self.ax.text(plot_location[0],plot_location[1]+0.03,0.0,fontsize=8)
                self.Q_labels[state].append(next_label)

    def _plot_arrows(self,Q_sa):
        if self.arrows is not None: 
            for arrow in self.arrows:
                arrow.remove() # Clear all previous arrows
        self.arrows=[]
        for state in range(self.n_states):
            plot_location = np.array(self._state_to_location(state)) + 0.5
            max_actions = full_argmax(Q_sa[state])
            for max_action in max_actions:
                new_arrow = arrow = Arrow(plot_location[0],plot_location[1],self.action_effects[max_action][0]*0.2,
                                          self.action_effects[max_action][1]*0.2, width=0.05,color='k')
                ax_arrow = self.ax.add_patch(new_arrow)
                self.arrows.append(ax_arrow)

def full_argmax(x):
    ''' Own variant of np.argmax, since np.argmax only returns the first occurence of the max '''
    return np.where(x == np.max(x))[0]            

def test():
    # Hyperparameters
    n_test_steps = 25
    step_pause = 0.5
    
    # Initialize environment and Q-array
    env = StochasticWindyGridworld()
    s = env.reset()
    Q_sa = np.zeros((env.n_states,env.n_actions)) # Q-value array of flat zeros

    # Test
    for t in range(n_test_steps):
        a = np.random.randint(4) # sample random action    
        s_next,r,done = env.step(a) # execute action in the environment
        p_sas,r_sas = env.model(s,a)
        print("State {}, Action {}, Reward {}, Next state {}, Done {}, p(s'|s,a) {}, r(s,a,s') {}".format(s,a,r,s_next,done,p_sas,r_sas))
        env.render(Q_sa=Q_sa,plot_optimal_policy=False,step_pause=step_pause) # display the environment
        if done: 
            s = env.reset()
        else: 
            s = s_next
    
if __name__ == '__main__':
    test()
