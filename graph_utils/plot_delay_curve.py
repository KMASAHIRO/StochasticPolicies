import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    delay_output_path = "min_delay.txt"
    fig_output_path = "temp_delay_curve.png"
    exp_dir = "./"
    map_name = "ingolstadt7"
    base = ["base_1", "base_2", "base_3", "base_4", "base_5"]
    delay_target_line = 30.7
    comparison = {
        "temp1.5":["temp1.5_1", "temp1.5_2", "temp1.5_3", "temp1.5_4", "temp1.5_5"],
        "temp2.0":["temp2.0_1", "temp2.0_2", "temp2.0_3", "temp2.0_4", "temp2.0_5"],
        "temp3.0":["temp3.0_1", "temp3.0_2", "temp3.0_3", "temp3.0_4", "temp3.0_5"]
    }

    min_delay_txt = ""

    base_delay_list = list()
    for x in base:
        delay_path = os.path.join(exp_dir, x, "avg_timeLoss.py")
        with open(delay_path) as f:
            delay = eval(f.readlines()[0].split(":")[1])[0]
        base_delay_list.append(delay)
    base_delay_mean = np.mean(base_delay_list, axis=0)
    base_delay_std = np.std(base_delay_list, axis=0)

    min_delay_txt += "base min (the best performance of the average curves): {:.2f}±{:.2f}".format(str(np.min(base_delay_mean)), str(base_delay_std[np.argmin(base_delay_mean)])) + "\n"
    min_delay_txt += "base min (the average of the best performances): {:.2f}±{:.2f}".format(str(np.mean(np.min(base_delay_list, axis=1))), str(np.std(np.min(base_delay_list, axis=1)))) + "\n"
    min_delay_txt += "base min (the best performance): {:.2f}".format(str(np.min(base_delay_list))) + "\n"
    
    fig_delay = plt.figure(figsize=(5,5), dpi=500)
    ax_delay = fig_delay.add_subplot(1, 1, 1)
    delay_steps = list(range(1, len(base_delay_mean)+1))
    ax_delay.plot(delay_steps, base_delay_mean, label="baseline")
    ax_delay.fill_between(delay_steps, base_delay_mean-base_delay_std, base_delay_mean+base_delay_std, alpha=0.3)
    for name in comparison.keys():
        x_delay_list = list()
        for x in comparison[name]:
            delay_path = os.path.join(exp_dir, x, "avg_timeLoss.py")
            with open(delay_path) as f:
                delay = eval(f.readlines()[0].split(":")[1])[0]
            x_delay_list.append(delay)
        
        x_delay_mean = np.mean(x_delay_list, axis=0)
        x_delay_std = np.std(x_delay_list, axis=0)
        min_delay_txt += x + " min (the best performance of the average curves): {:.2f}±{:.2f}".format(str(np.min(x_delay_mean)), str(x_delay_std[np.argmin(x_delay_mean)])) + "\n"
        min_delay_txt += x + " min (the average of the best performances): {:.2f}±{:.2f}".format(str(np.mean(np.min(x_delay_list, axis=1))), str(np.std(np.min(x_delay_list, axis=1)))) + "\n"
        min_delay_txt += x + " min (the best performance): {:.2f}".format(str(np.min(x_delay_list))) + "\n"

        delay_steps = list(range(1, len(x_delay_mean)+1))
        ax_delay.plot(delay_steps, x_delay_mean, label=name)
        ax_delay.fill_between(delay_steps, x_delay_mean-x_delay_std, x_delay_mean+x_delay_std, alpha=0.3)

    # literature value
    if delay_target_line is not None:
        target_values = [delay_target_line for i in range(len(base_delay_mean))]
        delay_steps = list(range(1, len(base_delay_mean)+1))
        ax_delay.plot(delay_steps, target_values, label="literature")

    ax_delay.legend()
    ax_delay.set_title("the average of the delay curves")
    ax_delay.set_xlabel("episodes")
    ax_delay.set_ylabel("Delays")

    fig_delay.tight_layout()
    fig_delay.savefig(fig_output_path, dpi=500)

    with open(delay_output_path, "w", encoding="utf-8") as f:
        f.write(min_delay_txt)
    
