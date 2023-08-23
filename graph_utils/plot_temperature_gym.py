import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    fig_output_path = "best_temperature_gym.png"
    exp_dir = "./"
    tasks = ["Acrobot-v1", "CartPole-v1"]
    comparison = [
        "temp0.0001", "temp0.001", "temp0.01", "temp0.1", "temp0.2", "temp0.4", "temp0.6", "temp0.8", 
        "temp1.0", "temp1.2", "temp1.4", "temp1.6", "temp1.8", "temp2.0", "temp2.2", "temp2.4", "temp2.6", 
        "temp2.8", "temp3.0", "temp3.2", "temp3.4", "temp3.6", "temp3.8", "temp4.0", "temp4.2", "temp4.4", 
        "temp4.6", "temp4.8", "temp5.0", "temp10.0", "temp100.0", "temp1000.0", "temp10000.0"
        ]

    steps = dict()
    for task_name in tasks:
        task_step = [[], []]
        for x in comparison:
            path = x + "/" + task_name + "_" + x + "_learncurve.csv"
            df = pd.read_csv(path)

            task_step[0].append(float(x.replace("temp","")))
            if task_name == "Acrobot-v1":
                task_step[1].append(np.min(df["mean_steps"].tolist()))
            elif task_name == "CartPole-v1":
                task_step[1].append(np.max(df["mean_steps"].tolist()))
        
        arg = np.argsort(task_step[0])
        task_step[0] = np.asarray(task_step[0])[arg]
        task_step[1] = np.asarray(task_step[1])[arg]
        
        steps[task_name] = task_step
    
    assert len(tasks) == 2
    fig, axes = plt.subplots(1,2, sharex="all", squeeze=False, constrained_layout=True, figsize=(10, 4), dpi=500)

    for i in range(len(tasks)):
        axes[0,i].scatter(steps[tasks[i]][0], steps[tasks[i]][1])
        axes[0,i].set_xlabel("Temperature")
        axes[0,i].set_xscale("log")
        axes[0,i].set_xlim(1e-5, 100000)

        if tasks[i] == "Acrobot-v1":
            axes[0,i].set_ylabel("Min steps")
        elif tasks[i] == "CartPole-v1":
            axes[0,i].set_ylabel("Max steps")
        
        axes[0,i].set_title(tasks[i])
    
    plt.savefig(fig_output_path, dpi=500)