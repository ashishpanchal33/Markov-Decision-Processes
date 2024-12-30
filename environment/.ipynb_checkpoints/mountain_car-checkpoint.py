# -*- coding: utf-8 -*-
"""
Author: John Mansfield
"""

import os
import warnings

import gym
import pygame
#from algorithms_rl import RL
#from algorithms_planner import Planner
#from test_env import TestEnv
import pickle
#import numpy as np

#env = gym.make('MountainCar-v0')




from gym.envs.classic_control import MountainCarEnv
from typing import List, Optional
from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional

import numpy as np

#import gymnasium as gym
#from gymnasium import Env, spaces, utils
#from gymnasium.envs.toy_text.utils import categorical_sample
#from gymnasium.error import DependencyNotInstalled
#from gymnasium.utils import seeding



import gym
from gym import spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled








#from gym.envs.classic_control import MountainCarEnv


class ExtendedMountainCarEnv(MountainCarEnv):
    #def reset_sp(self,state):
    #    self.state = state#your_very_special_method()
#
    #    #self.steps_beyond_done = None
    #    self._elapsed_steps =0
    #    return np.array(self.state, dtype=np.float32)

    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        state: Optional[list] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(options, -0.6, -0.4)
        
        #print(state)
        
        if type(state) != type(None):
            
            self.state = state
            #print('here',state)
        else:
            
            self.state = np.array([self.np_random.uniform(low=low, high=high), 0])
            #print(self.state)

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}    
    
    





    
    



#class _convert_state_obs_class(object):
#    def __init__(self,package_select):
#        self.package_select = package_select
#        
#    def 





        








class MountainCarEnv:
    def __init__(self, dim=5, max_episode_steps = 200,reward_threshold =-110.0, base = True):
        
        
        
        
        if base:
            self._env = gym.make('MountainCar-v0', render_mode=None,max_episode_steps=max_episode_steps)
        else:
            
            ExtendedMountainCarEnv_ref = gym.register(
                id="MountainCarEnv-v1_cust",
                entry_point=ExtendedMountainCarEnv,
            )


            #max_episode_steps = 200
            #env = gym.make('MountainCarEnv-v1_cust', render_mode=None,max_episode_steps=max_episode_steps)
            #env.spec.reward_threshold =-110.0










            self._env =gym.make('MountainCarEnv-v1_cust', render_mode=None,max_episode_steps=max_episode_steps)
            self._env.spec.reward_threshold =reward_threshold
            

        
        # Explanation of convert_state_obs lambda:
        # def function(state, done):
        # 	if done:
		#         return -1
        #     else:
        #         if state[2]:
        #             int(f"{state[0]+6}{(state[1]-2)%10}")
        #         else:
        #             int(f"{state[0]-4}{(state[1]-2)%10}")
                             
                             
                             
        
        #environment/MountainCar_envP_config.pkl
        
        self.dim__ = dim
        
        current_dir = os.path.dirname(__file__)
        file_name = 'MountainCar_envP_config.pkl'
        f = os.path.join(current_dir, file_name)
        
        try:
            self.packages = pickle.load(open(f, "rb"))
        except IOError:
            print("Pickle load failed.  Check path", f)
            
            
            
        try:
            package_select = self.packages[self.dim__]
            
            
            
            self._P = package_select['P']
        
        except IOError:
            print("Dim not pickled, check dim", dim)                             
                             
        
        #@staticmethod 
        #def _convert_state_obs_method_2(state,done,package_select = package_select):
        #    return (-1 if done else MountainCarEnv.mapper_state(
        #                                MountainCarEnv.discretize_state(state,
        #                                                                         package_select['low'],
        #                                                                        package_select['bin_width'],
        #                                                                        package_select['state_space']),
        #                                package_select['seriaized']))





        MountainCarEnv._convert_state_obs_method.__defaults__ = (package_select,)
        
        
        self._convert_state_obs = MountainCarEnv._convert_state_obs_method
        
        
        #self._convert_state_obs = lambda state, done: (
        #
        #-1 if done else MountainCarEnv.mapper_state(
        #                                MountainCarEnv.discretize_state(state,
        #                                                                         package_select['low'],
        #                                                                        package_select['bin_width'],
        #                                                                        package_select['state_space']),
        #                                package_select['seriaized'])
        #    
        #    
        #
        #
        #)
        
        
        #self._convert_state_obs = lambda state, done: (
        #    -1 if done else int(f"{state[0] + 6}{(state[1] - 2) % 10}") if state[2] else int(
        #        f"{state[0] - 4}{(state[1] - 2) % 10}"))
        # Transitions and rewards matrix from: https://github.com/rhalbersma/gym-blackjack-v1
        

        self._n_actions = self.env.action_space.n
        self._n_states = len(self._P)


     
    
    
    
    
        
    @staticmethod   
    def _convert_state_obs_method(state,done,package_select: Optional[dict] = None):
        
        
        
        
        return (-1 if done else MountainCarEnv.mapper_state(
                                        MountainCarEnv.discretize_state(state,
                                                                                 package_select['low'],
                                                                                package_select['bin_width'],
                                                                                package_select['state_space']),
                                        package_select['seriaized']))
        
        
        
        
    @staticmethod
    def mapper_state(state, seriaized):
        return np.where(
(seriaized == state).sum(axis =1) == 2)[0][0]


    @staticmethod # Discretize state
    def discretize_state(state,low,bin_width,state_space):
        
        
        
        state_adj = low + bin_width*((state - low)//bin_width)
        map_ = state_space[np.all(state_space.astype('float16') ==  (state_adj).astype('float16'),axis=2)]
        #print(state)
        state_adj = map_[0]
        return state_adj
        
        
        
    @property
    def n_actions(self):
        return self._n_actions

    @n_actions.setter
    def n_actions(self, n_actions):
        self._n_actions = n_actions

    @property
    def n_states(self):
        return self._n_states

    @n_states.setter
    def n_states(self, n_states):
        self._n_states = n_states

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, P):
        self._P = P

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, env):
        self._env = env

    @property
    def convert_state_obs(self):
        return self._convert_state_obs

    @convert_state_obs.setter
    def convert_state_obs(self, convert_state_obs):
        self._convert_state_obs = convert_state_obs


#if __name__ == "__main__":
#    MountainCarEnv = MountainCarEnv()

#    # VI/PI
#    # V, V_track, pi = Planner(MountainCarEnv.P).value_iteration()
#    # V, V_track, pi = Planner(MountainCarEnv.P).policy_iteration()

#    # Q-learning
#    Q, V, pi, Q_track, pi_track = RL(MountainCarEnv.env).q_learning(MountainCarEnv.n_states, MountainCarEnv.n_actions, MountainCarEnv.convert_state_obs)

#    test_scores = TestEnv.test_env(env=MountainCarEnv.env, render=True, pi=pi, user_input=False,
#                                   convert_state_obs=MountainCarEnv.convert_state_obs)