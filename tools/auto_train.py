import argparse
import configparser
import subprocess

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini", encoding='utf-8')

    train_path = config.get("DEFAULT", "do_train_path")
    map_dir = config.get("DEFAULT", "map_dir")
    
    map_name = config.get("DEFAULT", "map_name")
    device = config.get("DEFAULT", "device")

    sections = config.sections()
    experiments = dict()
    for sec in sections:
        if sec != "DEFAULT":
            experiments[sec] = dict(config.items(sec))
            if "device" not in experiments[sec].keys():
                experiments[sec]["device"] = device
            if "log_dir" not in experiments[sec].keys():
                experiments[sec]["log_dir"] = "./"
            if "reward_csv" not in experiments[sec].keys():
                experiments[sec]["reward_csv"] = map_name + "_IPPO_" + sec + "_reward.csv"
            if "run_name" not in experiments[sec].keys():
                experiments[sec]["run_name"] = "IPPO_" + sec

    for dir_name in experiments.keys():
        subprocess.run(["mkdir", dir_name])
        python_cmd = [
            "python", train_path, "--map_name", map_name, "--map_dir", map_dir
            ]
        
        for op,val in experiments[dir_name].items():
            if val == "True":
                op_cmd = ["--" + op]
            else:
                op_cmd = ["--" + op, val]
            python_cmd.extend(op_cmd)

        subprocess.Popen(python_cmd, cwd=dir_name)
