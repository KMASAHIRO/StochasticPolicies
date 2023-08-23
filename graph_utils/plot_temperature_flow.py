import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    fig_output_path = "best_temperature_flow.png"
    exp_dir = "./"
    delay_temp_flow = {
        "cologne1": {
            "1000": [],
            "2015": [],
            "3000": [],
            "4000": []
        },
        "cologne3": {
            "3000": [],
            "4494": [],
            "5000": [],
            "6000": []
        }, 
        "cologne8": {
            "1000": [],
            "2046": [],
            "3000": [],
            "4000": []
        }, 
        "ingolstadt1": {
            "1000": [],
            "1716": [],
            "2000": [],
            "3000": []
        }, 
        "ingolstadt7": {
            "2000": [],
            "3031": [],
            "4000": [],
            "5000": []
        }, 
        "ingolstadt21": {
            "3000": [],
            "4283": [],
            "5000": [],
            "6000": []
        }, 
        "arterial4x4": {
            "1000": [],
            "2484": [],
            "3000": [],
            "4000": []
        }, 
        "grid4x4": {
            "1000": [],
            "1473": [],
            "2000": [],
            "3000": []
        }
    }

    param_list = [
        "temp0.2", "temp0.4", "temp0.6", "temp0.8", "temp1.0", "temp1.2", "temp1.4", "temp1.6", "temp1.8", 
        "temp2.0", "temp2.2", "temp2.4", "temp2.6", "temp2.8", "temp3.0", "temp3.2", "temp3.4", "temp3.6", 
        "temp3.8", "temp4.0", "temp4.2", "temp4.4", "temp4.6", "temp4.8", "temp5.0", "temp6.0", "temp7.0", 
        "temp8.0", "temp9.0", "temp10.0", "temp20.0", "temp30.0", "temp40.0", "temp50.0", "temp60.0", 
        "temp70.0", "temp80.0", "temp90.0", "temp100.0", "temp200.0", "temp300.0", "temp400.0", "temp500.0", 
        "temp600.0", "temp700.0", "temp800.0", "temp900.0", "temp1000.0"
        ]
    
    for map_name in delay_temp_flow.keys():
        for flow in delay_temp_flow[map_name].keys():
            temp_list = list()
            min_delay_flow = list()
            for param in param_list:
                delay_path = os.path.join(exp_dir, map_name, flow, param, "avg_timeLoss.py")
                with open(delay_path) as f:
                    delay = eval(f.readlines()[0].split(":")[1])[0]
                min_delay_flow.append(np.min(delay))
                
                temp_list.append(float(param.replace("temp", "")))
            
            arg = np.argsort(temp_list)
            temp_list = np.asarray(temp_list)[arg]
            min_delay_flow = np.asarray(min_delay_flow)[arg]

            delay_temp_flow[map_name][flow].append(temp_list)
            delay_temp_flow[map_name][flow].append(min_delay_flow)
    
    maps = list(delay_temp_flow.keys())
    assert len(maps) == 8

    fig, axes = plt.subplots(2,4, sharex="all", constrained_layout=True, dpi=500, figsize=(15,5))
    for i in range(8):
      ax = axes[i//4, i%4]

      for flow in delay_temp_flow[maps[i]].keys():
        ax.scatter(delay_temp_flow[maps[i]][flow][0], delay_temp_flow[maps[i]][flow][1], label=flow)

      ax.set_ylabel("Delays")
      ax.set_title(maps[i].capitalize())
      if i>3:
        ax.set_xlabel("Temperature")
      ax.legend(loc='upper left', bbox_to_anchor=(1, 1)).get_frame().set_alpha(1.0)
      ax.set_xscale('log')

    plt.savefig(fig_output_path, dpi=500)
