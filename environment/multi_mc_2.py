import multiprocessing

import gym
#import gymnasium as gym
import pygame
from algorithms.rl import RL
#from algorithms.algorithms import RL

from environment.algorithms_rl import RL as RL_2


from examples.test_env import TestEnv

#from environment.test_env import TestEnv as TestEnv_2
import environment.frozen_lake as FL
from algorithms.planner import Planner
from environment.algorithms_planner import Planner as Planner_2

from environment.mountain_car import MountainCarEnv #as MountainCarEnv







import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from matplotlib.colors import LinearSegmentedColormap
        
        
from itertools import repeat, product

frozen_lake_ref = gym.register(
    id="Frozenlake-v1_cust",
    entry_point=FL.FrozenLakeEnv,
)




def create_n_map(map_seed = 1, count =10,
                env = FL,gym_configurations = {'id' : 'Frozenlake-v1_cust', 
                                                   'render_mode':None},
                 generate_random_map_config = dict(p =0.9),
                 size = 4,
                multiplier = 1000
                ):

    fl_map_list = []
    frozen_lake_list = []
    
    np.random.seed(map_seed)
    #random.seed(map_seed)
    
    for i in range(count):
        j = i*multiplier
        #np.random.seed(i)
        #random.seed(i)       
        
        
        #print(i)
        #size =  4
        #seed = 1

        fl_map = env.generate_random_map(size=size,
                                        
                                        **generate_random_map_config
                                        ,seed =j)
        #frozen_lake = gym.make('FrozenLake-v1', render_mode=None, desc=FL.generate_random_map(size=5))


        fl_map_list.append(fl_map)

        frozen_lake_list.append(  gym.make(**gym_configurations,
                               desc=fl_map,
                              #seed = 1
                               max_episode_steps = size*size*5#13
                              ))
        
        
    return fl_map_list,frozen_lake_list
        
        

def create_n_mc(map_seed = 1, count =10,
                env = MountainCarEnv,gym_configurations = {'id' : 'MountainCarEnv-v1_cust', 
                                                   'render_mode':None},
                 generate_random_map_config = dict(p =0.9),
                 size = 5,
                multiplier = 1000,
               
               base=False,
               max_episode_steps = 200,
                
                
                
                
               
               ):
    
    np.random.seed(map_seed)
    #random.seed(map_seed)
    


    env_list = []
    
    seed_list = [ map_seed+i*multiplier for i in range(count)]
    
    for i in seed_list:
        env_temp = env(dim=size,
                                    base =base,
                                      max_episode_steps=max_episode_steps)

        env_temp.env.reset(seed =i)

        env_list.append(env_temp)        

        
        
        
        
        
    return seed_list,env_list
        
        



def double(a):
    return a * 2

def driver_func(function=double, params = [(1, ), (2, ), (3, ), (4, )],PROCESSES=-1,
               
               type_ = 'apply_async'
               
               ):
    with multiprocessing.Pool(PROCESSES) as pool:
        #params = [(1, ), (2, ), (3, ), (4, )]
        #results = [pool.apply_async(function, kwds = p) for p in params]
        
        if type_ == 'apply_async':
            results = [pool.apply_async(function, kwds = p) for p in params]
            return [ r.get() for r in results]
        
        elif type_ == 'apply':
        
            results = [pool.apply(function, kwds = p) for p in params]

            #results = pool.map_async(function , params)


            #return [ r.get() for r in results]
            return results
        
        elif type_ == 'starmap':
            
            results = starmap_with_kwargs(pool, function, params) 
            
            return results
        
        
        elif type_ == 'starmap_async':
            
            results = starmap_with_kwargs_async(pool, function, params) 
            
            return results.get()

        
        elif type_ == 'starmap_additional':
            
            results = starmap_with_args_kwargs_async(pool, function, params[0],params[1]) 
            
            return results  
        
        
        
        
def starmap_with_kwargs(pool, fn, kwargs_iter):
    args_for_starmap = zip(repeat(fn,len(kwargs_iter)), kwargs_iter)
    return pool.starmap(apply_and_kwargs, args_for_starmap)



def starmap_with_kwargs_async(pool, fn, kwargs_iter):
    args_for_starmap = zip(list(repeat(fn,len(kwargs_iter))), kwargs_iter)
    return pool.starmap_async(apply_and_kwargs, args_for_starmap)


def apply_and_kwargs(fn, kwargs):
    return fn( **kwargs)
    

def starmap_with_args_kwargs_async(pool, fn, args_iter, kwargs_iter):
    
    
    args_for_starmap = zip(list(repeat(fn,len(kwargs_iter))),args_iter, kwargs_iter)
    return pool.starmap_async(apply_and_kwargs, args_for_starmap)


def apply_and_args_kwargs(fn,args, kwargs):
    return fn(*args, **kwargs)



    
    

#args_iter = zip(repeat(project_name), api_extensions)
#kwargs_iter = repeat(dict(payload={'a': 1}, key=True))
#branches = starmap_with_kwargs(pool, fetch_api, kwargs_iter)            
            
            
            

def mov_avg(x, w, axis = 0):
    
    return [x[m:m+w].mean(axis=0)  for m in range(len(x)-(w-1))]
    


def grid_values_heat_map(data,shape = (5,5),label = 'State Values',annot=False):
    #data = np.around(np.array(data).reshape(shape), 2)
    data = np.array(data).reshape(shape)
    
    
    df = pd.DataFrame(data=data)
    sns.heatmap(df, annot=annot,linewidths=0.1).set_title(label)
    plt.show()
    
 



    
    
    

def find_convergence(Q_track,window=100,threshold = 0.002,rl_ = False,index= 0):



    lis = np.array([ np.abs(i.max()) for i in  (Q_track - np.roll(Q_track, shift =1,axis =0))]
            )
    lis_2 = mov_avg(lis, w=100, axis = 0)

    #Q_track  = responses['train_responses'][0][3]
    #plt.plot([ np.abs(i.max()) for i in  (lis_2 - np.roll(lis_2, shift =1,axis =0))],

    #        )

    if rl_:
        return np.argmax(np.array(lis_2)<=threshold)


    
            
def Q_learning_modeling(env #= frozen_lake.env
                        , kwarg=dict(gamma = 0.99,
                                                             n_episodes=50000,
                                    
                                    
                                    window=100,threshold = 0.002
                                    
                                    ),
                       
                       RL =RL_2,
                       
                       
                       
                       
                       
                       
                       ):
    

    Q, V, pi, Q_track, pi_track,i = RL(env).q_learning(**kwarg)
    
    
    
    
    return (Q, V, pi, Q_track, pi_track,i 
            #find_convergence(Q_track,
            #    window=window,
            #    threshold = threshold,
            #    rl_ = True,
            #    index= 0)
           
           
           )








def _convert_state_obs_method(state,done,package_select=None):




    return (-1 if done else mapper_state(
                                discretize_state(state,
                                         package_select['low'],
                                        package_select['bin_width'],
                                        package_select['state_space']),
                                package_select['seriaized']))


def mapper_state(state, seriaized):
        return np.where(
(seriaized == state).sum(axis =1) == 2)[0][0]


def discretize_state(state,low,bin_width,state_space):
        
        
        
        state_adj = low + bin_width*((state - low)//bin_width)
        map_ = state_space[np.all(state_space.astype('float16') ==  (state_adj).astype('float16'),axis=2)]
        #print(state)
        state_adj = map_[0]
        return state_adj







































            
def Q_learning_modeling_mc(env #= frozen_lake.env
                        , kwarg=dict(gamma = 0.99,
                                                             n_episodes=50000,
                                    
                                    
                                    window=100,threshold = 0.002
                                    
                                    ),
                       
                       RL =RL_2,package_select=None
                       
                       
                       
                       
                       
                       
                       ):
    
    
    
    _convert_state_obs_method.__defaults__ = (package_select,)
    kwarg['convert_state_obs'] = _convert_state_obs_method    
    

    Q, V, pi, Q_track, pi_track,i = RL(env.env).q_learning(**kwarg)
    
    
    
    
    return (Q, V, pi, Q_track, pi_track,i 
            #find_convergence(Q_track,
            #    window=window,
            #    threshold = threshold,
            #    rl_ = True,
            #    index= 0)
           
           
           )






















    
def main_rl_mc(driver_func = driver_func,
                package_select=None,
         driver_func_config = dict(PROCESSES=-1,type_ = 'starmap_async'), 
         function_run = Q_learning_modeling_mc,
         learning_function_config = dict(
                      kwarg = dict(n_episodes=50000,gamma = 0.99,
                                window=100,threshold = 0.002)
         
         
         ),
               
               
               
               
         count_map_config =dict(map_seed = 1,
                                count =2,
             
                                env = MountainCarEnv,
                                gym_configurations = {'id' : 'MountainCarEnv-v1_cust', 
                                                   'render_mode':None},
                                 generate_random_map_config = dict(p =0.9),
                                 size = 4,
                                 multiplier = 1000,
                               
                                base =False,
                                max_episode_steps=200
                               
                               ),
           
           
           map_list =[], env_list = [],
           
           create_new_map_flag = False
            
           )  :

    
    
    
    
 
    #learning_function_config['kwarg']['ns'] = env_list[0].n_states
    

    
    
    
    
    
    
    
    
    
    
    size = count_map_config['size']
    if (not create_new_map_flag ) or (len(env_list) == 0):
        map_list,env_list = create_n_mc(**count_map_config)

        
        
    learning_function_config['kwarg']['nS'] = env_list[0].n_states
    learning_function_config['kwarg']['nA'] = env_list[0].n_actions
        

    
    
    
    #print(_convert_state_obs_method(np.array([0.1,0.1]),False))
    
    
    params = [ dict(env = env_list[i]
                    ,RL=RL_2
                    ,**learning_function_config,package_select=package_select) 
              
                    for i in range(len(map_list))]
 

    
    return {'train_responses':driver_func(function_run,params = params,**driver_func_config),              
            'map_list':map_list, 
                
            
            'env_list' :env_list}    
   
    

    
    
    
    
        
    
def main_rl(driver_func = driver_func,
         driver_func_config = dict(PROCESSES=-1,type_ = 'starmap_async'), 
         function_run = Q_learning_modeling,
         learning_function_config = dict(kwarg = dict(n_episodes=50000,gamma = 0.99,
                                                          window=100,threshold = 0.002)),
         count_map_config =dict(map_seed = 1,
                                count =2,
             
                                env = FL,
                                gym_configurations = {'id' : 'Frozenlake-v1_cust', 
                                                   'render_mode':None},
                                 generate_random_map_config = dict(p =0.9),
                                 size = 4,
                                 multiplier = 1000
                               
                               
                               
                               ),
           
           
           map_list =[], env_list = [],
           
           create_new_map_flag = False
            
           )  :
    
    
    
    
    
    #generate_random_map_config['size'] = 
    
    
    if (not create_new_map_flag ) or (len(env_list) == 0):
    
        map_list, env_list = create_n_map(**count_map_config)
        
    size = count_map_config['size']
    #gym_configurations['max_episode_steps'] = size*size*5
    
    #print(env_list)
    
    params = [ dict(env = env_list[i]
                    ,RL=RL_2
                    ,**learning_function_config) 
              
                    for i in range(len(map_list))]
 

    #print(params)


    #Q, V, pi, Q_track, pi_track = Q_learning_modeling(env = frozen_lake.env,,RL =RL_2,
    #                                                  kwarg = dict(n_episodes=50
    #                                                      ))
    
    return {'train_responses':driver_func(function_run,params = params,**driver_func_config),              
            'map_list':map_list, 
                
            
            'env_list' :env_list}
    
    
    

            
def PIVI_modeling(env=FL #= frozen_lake.env
                        , kwarg=dict(n_iters=2000,gamma=0.99),
                       
                       planner_ =Planner_2,vi=True):
    

    plan_fl = planner_(env.env.P)


    
    if vi:
        V, V_track, pi,pi_track,converge_index = plan_fl.value_iteration(
            **kwarg)
    
    else:
        V, V_track, pi,pi_track,converge_index = plan_fl.policy_iteration(
            **kwarg)
        



    
    
    
    
    #Q, V, pi, Q_track, pi_track = RL(env).q_learning(**kwarg)
    return V, V_track, pi,pi_track,converge_index    




            
def PIVI_modeling_mc(env=MountainCarEnv #= frozen_lake.env
                        , kwarg=dict(n_iters=2000,gamma=0.99),
                       
                       planner_ =Planner_2,vi=True):
    

    plan_fl = planner_(env.P)


    
    if vi:
        V, V_track, pi,pi_track,converge_index = plan_fl.value_iteration(
            **kwarg)
    
    else:
        V, V_track, pi,pi_track,converge_index = plan_fl.policy_iteration(
            **kwarg)
        



    
    
    
    
    #Q, V, pi, Q_track, pi_track = RL(env).q_learning(**kwarg)
    return V, V_track, pi,pi_track,converge_index    











def main_pivi(driver_func = driver_func,
         driver_func_config = dict(PROCESSES=-1,type_ = 'starmap_async'), 
         function_run = PIVI_modeling,
         learning_function_config = dict(kwarg = dict(n_iters=2000,gamma=0.99,
                                                          ),vi = True),
         count_map_config =dict(map_seed = 1,
                                count =2,
             
                                env = FL,
                                gym_configurations = {'id' : 'Frozenlake-v1_cust', 
                                                   'render_mode':None},
                                 generate_random_map_config = dict(p =0.9),
                                 size = 4,
                                 multiplier = 1000
                               
                               
                               
                               ),
           
           
           map_list =[], env_list = [],
           
           create_new_map_flag = False
            
           )  :
    
    
    
    
    
    #generate_random_map_config['size'] = 
    
    
    if (not create_new_map_flag ) or (len(env_list) == 0):
    
        map_list, env_list = create_n_map(**count_map_config)
        
    size = count_map_config['size']
    #gym_configurations['max_episode_steps'] = size*size*13
    
    #print(env_list)
    
    params = [ dict(env = env_list[i],planner_=Planner_2,**learning_function_config) 
              for i in range(len(map_list))]
 

    #print(params)


    #Q, V, pi, Q_track, pi_track = Q_learning_modeling(env = frozen_lake.env,,RL =RL_2,
    #                                                  kwarg = dict(n_episodes=50
    #                                                      ))
    
    return {'train_responses':driver_func(function_run,params = params,**driver_func_config),              
            'map_list':map_list, 
                
            
            'env_list' :env_list}
    

    
    
    
def main_pivi_mc(driver_func = driver_func,
         driver_func_config = dict(PROCESSES=-1,type_ = 'starmap_async'), 
         function_run = PIVI_modeling_mc,
         learning_function_config = dict(kwarg = dict(n_iters=2000,gamma=0.99,
                                                          ),vi = True),
         count_map_config =dict(map_seed = 1,
                                count =2,
             
                                env = FL,
                                gym_configurations = {'id' : 'MountainCarEnv-v1_cust', 
                                                   'render_mode':None},
                                 generate_random_map_config = dict(p =0.9),
                                 size = 4,
                                 multiplier = 1000,
                                
                                
                                base =False,
                                max_episode_steps=200
                               
                               
                               
                               ),
           
           
           map_list =[], env_list = [],
           
           create_new_map_flag = False
            
           )  :
    
    
    
    
    
    #generate_random_map_config['size'] = 
    
    
    
    
    size = count_map_config['size']
    if (not create_new_map_flag ) or (len(env_list) == 0):
        map_list,env_list = create_n_mc(**count_map_config)
            
            
        
    
    
    params = [ dict(env = env_list[i],planner_=Planner_2,**learning_function_config) 
              for i in range(len(map_list))]
 

    return {'train_responses':driver_func(function_run,params = params,**driver_func_config),              
            'map_list':map_list, 
                
            
            'env_list' :env_list}
    