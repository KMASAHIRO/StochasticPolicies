# StochasticPolicies  
Source code for "Rethinking the Stochastic Policy Gradient Methods Using a Traffic Simulator"  
This source code is based on [RESCO](https://github.com/Pi-Star-Lab/RESCO).  
  
PPO agents with stochastic policies can be trained for traffic signal control tasks. [Simulation for Urban Mobility (SUMO)](https://eclipse.dev/sumo/) is used as a traffic simulator.  
REINFORCE agents with stochastic policies can be trained for some [OpenAI Gym](https://www.gymlibrary.dev/) tasks.

## Installation  
The modules can be installed using pip with the command below.  
```
$pip install git+https://github.com/KMASAHIRO/StochasticPolicies
```
The required packages in "requirements.txt" will be installed automatically.  
Using SUMO internally, so [SUMO_HOME](https://sumo.dlr.de/docs/Basics/Basic_Computer_Skills.html#sumo_home) environment variable must be set.  
If you install SUMO in [other ways](https://sumo.dlr.de/docs/Downloads.php), it will be easier to set the environment variable.

## Using Stochastic Policies
Four types of stochastic policies can be used.  
