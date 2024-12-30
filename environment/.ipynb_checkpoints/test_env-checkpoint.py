# -*- coding: utf-8 -*-
"""
Author: John Mansfield

documentation added by: Gagandeep Randhawa
"""

"""
Simulation of the agent's decision process after it has learned a policy.
"""
import gym
import pygame
import numpy as np
import time
import random

from itertools import repeat
from typing import List, Optional


def convert_state_obs(state, done):
    return state 

class TestEnv_2:
    def __init__(self):
        
        
        
        
        pass

    
    
    @staticmethod
    def test_env_inner_loop(env,convert_state_obs,n_actions,
                           
                           
                           render=None, n_iters=10, pi=None, 
                            user_input=False,
                           
                           seed: Optional[int] = None,
                             package_select: Optional[dict] = None
                           
                           
                           ):
        state, info = env.reset()#seed)
        done = False
        state = convert_state_obs(state, done)
        total_reward = 0
        total_steps = 0
        start_time = time.time()
        while not done:
            total_steps +=1
            if user_input:
                # get user input and suggest policy output
                print("state is %i" % state)
                print("policy output is %i" % pi[state])
                while True:
                    action = input("Please select 0 - %i then hit enter:\n" % int(n_actions-1))
                    try:
                        action = int(action)
                    except ValueError:
                        print("Please enter a number")
                        continue
                    if 0 <= action < n_actions:
                        break
                    else:
                        print("please enter a valid action, 0 - %i \n" % int(n_actions - 1))
            else:
                action = pi[state]
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = convert_state_obs(next_state, done)
            state = next_state
            total_reward = reward + total_reward
            #print(total_reward)

            
            
        return total_reward,total_steps, (time.time() -start_time    )
        #test_steps[i] = total_steps
        #test_scores[i] = total_reward
#
        #test_total_time[i] = time.time() -start_time    
    
    
    
    @staticmethod
    def test_env(env, render=None, n_iters=10, pi=None, 
                 user_input=False, convert_state_obs=lambda state, done: state,seed =1, package_select: Optional[dict] = None):
        """
        Parameters
        ----------------------------
        env {OpenAI Gym Environment}:
            MDP problem

        render {Boolean}:
            openAI human render mode

        n_iters {int}, default = 10:
            Number of iterations to simulate the agent for

        pi {lambda}:
            Policy used to calculate action value at a given state

        user_input {Boolean}:
            Prompt for letting user decide which action to take at a given state

        convert_state_obs {lambda}:
            The state conversion utilized in BlackJack ToyText problem.
            Returns three state tuple as one of the 280 converted states.


        Returns
        ----------------------------
        test_scores {list}:
            Log of reward at the end of each iteration
        """
        if render:
            # unwrap env and and reinit in 'human' render_mode
            env_name = env.unwrapped.spec.id
            
            try:
                
            
                env = gym.make(env_name, render_mode=render#'human'
                          
                          )
            except Exception as e:
                print(e)
                error()
            
            
            
            
        np.random.seed(seed)    
        random.seed(seed)
        
        
        n_actions = env.action_space.n
        test_scores = np.full([n_iters], np.nan)
        
        test_steps = np.full([n_iters], np.nan)
        
        test_total_time = np.full([n_iters], np.nan)
        env.reset(seed=seed)
        
        for i in range(0, n_iters):
            #env.reset(seed=i)
            (test_scores_value, 
             test_steps_value,
             test_total_time_value) = TestEnv_2.test_env_inner_loop(
                                    env=env,
                                    convert_state_obs=convert_state_obs,
                                    n_actions=n_actions,
                                    render=render, 
                                    n_iters=n_iters,
                                    pi=pi, user_input=user_input,seed = None
                                    ,package_select=package_select
                           )
            test_scores[i], test_steps[i],test_total_time[i] =(test_scores_value, test_steps_value,test_total_time_value)
        env.close()
        return test_scores, test_steps,test_total_time
    
   
    
    
    @staticmethod
    def test_env_parallel(env, parellel_func,render=None, n_iters=10,
                          pi=None, user_input=False, 
                          convert_state_obs=convert_state_obs,#,
                         
                         PROCESSES = -1,type_ = 'apply_async'
                         ):
        """
        Parameters
        ----------------------------
        env {OpenAI Gym Environment}:
            MDP problem

        render {Boolean}:
            openAI human render mode

        n_iters {int}, default = 10:
            Number of iterations to simulate the agent for

        pi {lambda}:
            Policy used to calculate action value at a given state

        user_input {Boolean}:
            Prompt for letting user decide which action to take at a given state

        convert_state_obs {lambda}:
            The state conversion utilized in BlackJack ToyText problem.
            Returns three state tuple as one of the 280 converted states.


        Returns
        ----------------------------
        test_scores {list}:
            Log of reward at the end of each iteration
        """
        if render:
            # unwrap env and and reinit in 'human' render_mode
            env_name = env.unwrapped.spec.id
            
            try:
                
            
                env = gym.make(env_name, render_mode=render#'human'
                          
                          )
            except Exception as e:
                print(e)
                error()
            
        
        
        seed_list = list(range(n_iters))
        
        
        n_actions = env.action_space.n
        test_scores = np.full([n_iters], np.nan)
        
        test_steps = np.full([n_iters], np.nan)
        
        test_total_time = np.full([n_iters], np.nan)
        
        #env.reset(seed=seed_list[0])
        
        kwargs_list = list(repeat(dict(env=env,
                                    convert_state_obs=convert_state_obs,
                                    n_actions=n_actions,
                                    render=render, 
                                    n_iters=n_iters,
                                    pi=pi, user_input=user_input), n_iters))
        
        
        for i in seed_list:
            kwargs_list[i]['seed'] = i*1000
            kwargs_list[i]['env'].reset(seed = i*1000)
            
            
             
            
            
            
        #print(kwargs_list)

        
        
        #test_scores, test_steps,test_total_time 
        response= parellel_func(function=TestEnv_2.test_env_inner_loop, params = kwargs_list,PROCESSES=PROCESSES,
               
               type_ =type_ 
               
               )
        
        
        
        
        #for i in range(0, n_iters):
        #    (test_scores_value, 
        #     test_steps_value,
        #     test_total_time_value) = test_env_inner_loop(
        #                            env=env,
        #                            convert_state_obs=convert_state_obs,
        #                            n_actions=n_actions,
        #                            render=render, 
        #                            n_iters=n_iters,
        #                            pi=pi, user_input=user_input,
        #                   )
        #    test_scores[i], test_steps[i],test_total_time[i] =(test_scores_value, test_steps_value,test_total_time_value)
        env.close()
        return response