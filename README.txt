This will outline how to retrieve your code, how to retrieve your data, how to retrieve the dependencies (libraries) for running your code, and how to run your code.

# Assignment Details
- @uthor: Ashish Panchal
- GTuser : apanchal33
- Designation: Student- OMSCS Georgia tech
- email : apanchal33@gatech.edu
- Subject : CS7641- Machine learning


# Title : Instructions for implementing Code for Assignment 3 – Unsupervised learning and Dimensionality Reduction
This document details the steps for retrieval of code, data and dependencies, followed by steps for execution of the code.

# The code has directory structure, as follows:







- Assignment 3_main: this is the main folder of the project, containing the following:

    1. File: Frozen lake Training and data creation.ipynb : main training file for VI,PI and Q learning over frozen lake , contain, function to run over grid of parameters and also save them
    2. File: Discretizer_Mountain_car.ipynb: contains functions to descritize the the Mountain car statespace and generate transition matrix for different sizes
    3. File: mountain_car_Training and_data_creation.ipynb : main training file for VI,PI and Q learning over Mountain car , contain, function to run over grid of parameters and also save them
    4. File: Evaluations.ipynb : Test evaluation for Moutain car and frozen lake, PI,VI and q learning
    5. File: Summary_plots and data.ipynb : generate summary charts from evaluations, size comparisons, convergence and exploration-exploitation tradeoff
    
    6. Folder: environment : contains all the utility modules:

        - algorithms_planner.py        : algorithm files contains modeuls for training PI and VI
        - algorithms_rl.py            : algorithm files contains modeuls for training Q learning and sarsa
        - img/                        : requirements for frozen lake
        - font/                         : requirements for frozen lake
        - frozen_lake.py               : custom froen lake env.
        - mountain_car.py              : custom mountain car env 
        - MountainCar_envP_config.pkl   : mountain car state transition matrix and other config
        - multi_.py                    : Utility function to run training using multi processing
        - multi_mc_2.py                  : Utility function to run training using multi processing adapted for custom mountain car 
        - plots_.py                      : plot example utility functions 
        - test_env.py                    : Testing modules           

        
        
        
    2. Folder: Figure: this folder the base location for saving plots
    3. Folder: model_bests : contains model summaries
    4. Folder: analysis db: contains analysis and evaluation chart csv.
    5. File: apanchal33-analysis.pdf : Assignment 4 report
    7. File: requirements.txt : lists requirements for the code
    8. ADA: this folder contains intermediate analytical dataset for each analysis. however could not be uploaded

## 1. Retrieval of the Code:
The code is saved at Sharable Box location : https://gatech.box.com/s/zf9hp2iq017opy8od4sgmz0eax0bwu86

## 2. Retrieval of the Dependencies:
The code uses particular version of different python packages,
These packages with their versions are listed in the requirements.txt file.
Please execute the installation using : ‘pip install -r requirments.txt’ using preferred command line tool or environment from “Assignment_4_main” location.  

## 3. Execution of code:
The code requires a specific order of execution:

    1. Frozen lake Training and data creation.ipynb
    2. Discretizer_Mountain_car.ipynb
    3. mountain_car_Training and_data_creation.ipynb
    4. Evaluations.ipynb
    5. Summary_plots and data.ipynb

Each code notebook It contains all the required function definitions used, And can be executed in a serialized way from first cell to the last.

The code is divided in to 4 parts

1. (generating transition matrix for MC)
2.  Training of planing algorithms over a grid for differnt algorithms
3.  Saving training data, V,P,Q,P_track,V_track and convergence points, environment intances , maps and seeds
4.  using these saved data for testing and generate evaluation measures
5.  use the above measures and training data to perform analysis: comparitive, convergence and exploration exploitation



 Kindly node, the code is provided with markdown headings and sections, to make best use of the configuration, please use and IDE which creates a table of content based on the same. (jupyterlab, jupyterbook with TOC ext.)

            
            
