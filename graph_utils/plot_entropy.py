import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    fig_output_path = "temp_entropy.png"
    exp_dir = "./"
    maps = ["cologne1", "cologne3", "cologne8", "ingolstadt1", "ingolstadt7", "ingolstadt21", "arterial4x4", "grid4x4"]
    comparison = {
        "1.0":[["base_1", "base_2", "base_3", "base_4", "base_5"], ["base_no_entropy_1", "base_no_entropy_2", "base_no_entropy_3", "base_no_entropy_4", "base_no_entropy_5"]],
        "1.5":[["temp1.5_1", "temp1.5_2", "temp1.5_3", "temp1.5_4", "temp1.5_5"], ["temp1.5_no_entropy_1", "temp1.5_no_entropy_2", "temp1.5_no_entropy_3", "temp1.5_no_entropy_4", "temp1.5_no_entropy_5"]],
        "2.0":[["temp2.0_1", "temp2.0_2", "temp2.0_3", "temp2.0_4", "temp2.0_5"], ["temp2.0_no_entropy_1", "temp2.0_no_entropy_2", "temp2.0_no_entropy_3", "temp2.0_no_entropy_4", "temp2.0_no_entropy_5"]],
        "3.0":[["temp3.0_1", "temp3.0_2", "temp3.0_3", "temp3.0_4", "temp3.0_5"], ["temp3.0_no_entropy_1", "temp3.0_no_entropy_2", "temp3.0_no_entropy_3", "temp3.0_no_entropy_4", "temp3.0_no_entropy_5"]]
    }

    delays = dict()
    for map_name in maps:
        entropy_delay = [[], []]
        no_entropy_delay = [[], []]
        for key in comparison:
            entropy_exp = comparison[key][0]
            no_entropy_exp = comparison[key][1]

            tmp = list()
            for x in entropy_exp:
                delay_path = os.path.join(exp_dir, map_name, x, "avg_timeLoss.py")
                with open(delay_path) as f:
                    delay = eval(f.readlines()[0].split(":")[1])[0]
                tmp.append(delay)
            tmp_mean = np.mean(tmp, axis=0)
            tmp_std = np.std(tmp, axis=0)
            
            entropy_delay[0].append(np.min(tmp_mean))
            entropy_delay[1].append(tmp_std[np.argmin(tmp_mean)])

            tmp = list()
            for x in no_entropy_exp:
                delay_path = os.path.join(exp_dir, map_name, x, "avg_timeLoss.py")
                with open(delay_path) as f:
                    delay = eval(f.readlines()[0].split(":")[1])[0]
                tmp.append(delay)
            tmp_mean = np.mean(tmp, axis=0)
            tmp_std = np.std(tmp, axis=0)
            
            no_entropy_delay[0].append(np.min(tmp_mean))
            no_entropy_delay[1].append(tmp_std[np.argmin(tmp_mean)])
        
        delays[map_name] = {"entropy": entropy_delay, "no_entropy": no_entropy_delay}
    
    assert len(maps) == 8
    fig, axes = plt.subplots(2,4, sharex="all", constrained_layout=True, dpi=500, figsize=(15,5))
    bar_width = 0.2
    capsize = 3
    for i in range(8):
        ax = axes[i//4, i%4]
        x = [j+1 for j in range(len(list(comparison.keys())))]
        y = [delays[maps[i]]["entropy"][0][j] for j in range(len(list(comparison.keys())))]
        yerr = [delays[maps[i]]["entropy"][1][j] for j in range(len(list(comparison.keys())))]
        ax.bar(np.asarray(x)-bar_width/2, np.asarray(y), yerr=np.asarray(yerr), width=bar_width, capsize=capsize)

        x_no_entropy = [j+1 for j in range(len(list(comparison.keys())))]
        y_no_entropy = [delays[maps[i]]["no_entropy"][0][j] for j in range(len(list(comparison.keys())))]
        yerr_no_entropy = [delays[maps[i]]["no_entropy"][1][j] for j in range(len(list(comparison.keys())))]
        ax.bar(np.asarray(x_no_entropy)+bar_width/2, np.asarray(y_no_entropy), yerr=np.asarray(yerr_no_entropy), width=bar_width, capsize=capsize)
        
        ax.set_ylabel("Delays")
        ax.set_title(maps[i].capitalize())
        if i>3:
            ax.set_xlabel("Temperature")
            plt.xticks(x, list(comparison.keys()))
    
    plt.savefig(fig_output_path, dpi=500)


