# StochasticPolicies  
Source code for "Rethinking Stochastic Policy Gradient Methods Using Traffic Simulator Benchmark".  
This source code is based on [RESCO (Reinforcement Learning Benchmarks for Traffic Signal Control)](https://github.com/Pi-Star-Lab/RESCO).  
  
PPO agents with stochastic policies can be trained for traffic signal control tasks. [SUMO (Simulation for Urban Mobility)](https://eclipse.dev/sumo/) is used as a traffic simulator.  
REINFORCE agents with stochastic policies can be trained for some [OpenAI Gym](https://www.gymlibrary.dev/) tasks.

## Installation  
The modules can be installed using pip with the command below.  
```
$pip install git+https://github.com/KMASAHIRO/StochasticPolicies
```
The required packages in "requirements.txt" will be installed automatically.  
Using SUMO internally, so [SUMO_HOME](https://sumo.dlr.de/docs/Basics/Basic_Computer_Skills.html#sumo_home) environment variable must be set.  
If you install SUMO in [other ways](https://sumo.dlr.de/docs/Downloads.php), it will be easier to set the environment variable.

## Repository detail
"StochasticPolicies" contains the modules for training.  
"environments" contains traffic map data of SUMO.  
"route_change" contains traffic map data of SUMO, and those traffic volumes are different from "environments".  
"tools" contains modules to easily perform the training.  
do_train.py under "tools" directory is used for training.  
With auto_train.py and config.ini, multiple trainings can be easily performed at the same time.  

## How to train
Four types of stochastic policies can be used.  
  
The first one is setting a temperature parameter in the softmax function of the output layer.
```
$python do_train.py --run_name temperature_training --map_name cologne1 --map_dir ./ --temperature 3.0
```
run_name is the name of the training, and you can name it freely.  
map_name is the name of the traffic map, and you can choose from cologne1, cologne3, cologne8, ingolstadt1, ingolstadt7, ingolstadt21, arterial4x4, and grid4x4.  
map_dir is the path to the directory that contains "environments".  
These three options must be specified for every training.  
temperature is the temperature parameter you set.  

The second one is setting a Gaussian noise to the output of the middle layer.  
```
$python do_train.py --run_name noise_training --map_name cologne1 --map_dir ./ --noise 0.02
```
noise is the standard deviation of the Gaussian distribution.  

The third one is using [NoisyNet](https://github.com/Kaixhin/NoisyNet-A3C) as the output layer.  
```
$python do_train.py --run_name noisy_training --map_name cologne1 --map_dir ./ --layer_type noisy
```

The fourth one is using [Bayes by Backprop (BBB)](https://github.com/nitarshan/bayes-by-backprop/blob/master/Weight%20Uncertainty%20in%20Neural%20Networks.ipynb) as the output layer.  
```
$python do_train.py --run_name bbb_training --map_name cologne1 --map_dir ./ --layer_type bbb
```

Delay is recorded in avg_timeLoss.py after training.  
Delay is the primary metric for training, and the lower the better.  

## LICENSE
SUMO scenarios are provided in the environment and route_change directories. All scenarios are distributed under the original license. Information on the Cologne scenario can be found at (https://sumo.dlr.de/docs/Data/Scenarios/TAPASCologne.html). Information on the Ingolstadt scenario can be found at (https://github.com/silaslobo/InTAS). Information on the other scenarios can be found at (https://sumo.dlr.de/docs/Data/Scenarios.html).  

## Results
The graph below shows change in the minimum value of delay due to the temperature parameter.  
![log2x4](https://github.com/KMASAHIRO/StochasticPolicies/assets/74399610/0c8b5f39-a3d9-4425-8525-ee02461b7780)

The graph below shows changes in the effect of the temperature parameter due to changes in traffic volume (the legend represents traffic volume (vehicles/hour). Orange represents the reference traffic volume, which represents the same value as the graph above.).  
![traffic_flow](https://github.com/KMASAHIRO/StochasticPolicies/assets/74399610/25a588be-f5e5-4d9e-ba4d-e1d1806808ba)

## Citation
```
@inproceedings{kato2024reconsidering,
title = {Reconsidering Stochastic Policy Gradient Methods for Traffic Signal Control},
author = {Kato, Masahiro and Kojima, Ryosuke},
booktitle = {Advances and Trends in Artificial Intelligence. Theory and Applications: 37th International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems, IEA/AIE 2024},
pages = {442–453},
year = {2024},
url = {https://doi.org/10.1007/978-981-97-4677-4_36},
doi = {10.1007/978-981-97-4677-4_36},
isbn = {978-981-97-4676-7},
publisher = {Springer-Verlag},
month={Jul. 11},
place = {Hradec Kralove, Czech Republic}
}
```
