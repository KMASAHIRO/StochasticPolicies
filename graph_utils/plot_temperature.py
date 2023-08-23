import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    fig_output_path = "best_temperature.png"
    exp_dir = "./"
    maps = ["cologne1", "cologne3", "cologne8", "ingolstadt1", "ingolstadt7", "ingolstadt21", "arterial4x4", "grid4x4"]
    comparison = [
        "temp0.2", "temp0.4", "temp0.6", "temp0.8", "temp1.2", "temp1.4", "temp1.6", "temp1.8", "temp2.2", 
        "temp2.4", "temp2.6", "temp2.8", "temp3.2", "temp3.4", "temp3.6", "temp3.8", "temp4.0", "temp4.2", 
        "temp4.4", "temp4.6", "temp4.8", "temp5.0", "temp10.0", "temp100.0", "temp1000.0", "temp10000.0"
        ]
    base = {
        "1.0":["base_1", "base_2", "base_3", "base_4", "base_5"],
        "1.5":["temp1.5_1", "temp1.5_2", "temp1.5_3", "temp1.5_4", "temp1.5_5"],
        "2.0":["temp2.0_1", "temp2.0_2", "temp2.0_3", "temp2.0_4", "temp2.0_5"],
        "3.0":["temp3.0_1", "temp3.0_2", "temp3.0_3", "temp3.0_4", "temp3.0_5"]
    }

    delays = dict()
    for map_name in maps:
        map_delay = [[], []]
        for x in comparison:
            delay_path = os.path.join(exp_dir, map_name, x, "avg_timeLoss.py")
            with open(delay_path) as f:
                delay = eval(f.readlines()[0].split(":")[1])[0]
            
            temperature = float(x.replace("temp",""))
            map_delay[0].append(temperature)
            map_delay[1].append(np.min(delay))
        
        for temperature in base:
            tmp = list()
            for x in base[temperature]:
                delay_path = os.path.join(exp_dir, map_name, x, "avg_timeLoss.py")
                with open(delay_path) as f:
                    delay = eval(f.readlines()[0].split(":")[1])[0]
                tmp.append(delay)
            
            map_delay[0].append(float(temperature))
            map_delay[1].append(np.mean(np.min(tmp, axis=1)))
        
        arg = np.argsort(map_delay[0])
        map_delay[0] = np.asarray(map_delay[0])[arg]
        map_delay[1] = np.asarray(map_delay[1])[arg]
        
        delays[map_name] = map_delay
    
    assert len(maps) == 8
    fig, axes = plt.subplots(2,4, sharex="all", squeeze=False, constrained_layout=True, figsize=(15,5), dpi=500)
    
    for i in range(8):
      ax = axes[i//4, i%4]
      ax.scatter(delays[maps[i]][0], delays[maps[i]][1])
      ax.set_title(maps[i].capitalize())
      if i>3:
        ax.set_xlabel("Temperature")
      ax.set_xscale("log")
      ax.set_ylabel("Delays")
      ax.set_xlim(1e-1,100000)
    plt.savefig(fig_output_path, dpi=500)