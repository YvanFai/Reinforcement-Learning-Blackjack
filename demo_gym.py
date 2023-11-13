from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from tqdm import tqdm
import gymnasium as gym

env = gym.make('Blackjack-v1' ,  sab = True , render_mode = "rgb_array")

#Reset the environment to get first observation
done = False
observation , info = env.reset()

#Action
action = env.action_space.sample()

#Information received after action
observation , reward , terminated , truncated , info = env.step(action)

#Agent
class BlackjackAgent():
    def __init__(self , 
                 learning_rate : float , 
                 initial_epsilon : float , 
                 epsilon_decay : float , 
                 final_epsilon : float , 
                 discount_factor : 0.95):
        
        #Initilize q 
        self.q_values = defaultdict(lambda : np.zeros(env.action_space))

        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        #error tracking
        self.training_error = []

    def get_action(self , obs : [int , int , bool])-> int:

        #Epsilon greedy
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))   

    
    def update(self ,
               obs : [int , int , bool] ,
               action : int ,
               reward : float ,
               terminated : bool , 
               next_obs : [int , int , bool]):
        
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_diff = reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        self.q_values[obs][action] = self.q_values[obs][action] + self.lr * temporal_diff
        
        self.training_error.append(temporal_diff)

        

        






















