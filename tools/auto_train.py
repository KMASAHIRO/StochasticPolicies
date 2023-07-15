import os
import argparse
import configparser
import subprocess

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini", encoding='utf-8')

    train_path = config.get("DEFAULT", "do_train_path")
    map_name = config.get("DEFAULT", "map_name")

    sections = config.sections()
    experiments = dict()
    for sec in sections:
        if sec != "DEFAULT":
            experiments[sec] = dict(config.items(sec))
            experiments[sec].pop("do_train_path")
            if "log_dir" not in experiments[sec].keys():
                experiments[sec]["log_dir"] = "./"
            if "reward_csv" not in experiments[sec].keys():
                experiments[sec]["reward_csv"] = map_name + "_IPPO_" + sec + "_reward.csv"
            if "run_name" not in experiments[sec].keys():
                experiments[sec]["run_name"] = "IPPO_" + sec
            
            experiments[sec]["map_dir"] = os.path.abspath(experiments[sec]["map_dir"])

    for dir_name in experiments.keys():
        subprocess.run(["mkdir", dir_name])
        python_cmd = [
            "python", os.path.abspath(train_path)
            ]
        
        for op,val in experiments[dir_name].items():
            if val == "True":
                op_cmd = ["--" + op]
            else:
                op_cmd = ["--" + op, val]
            python_cmd.extend(op_cmd)

        subprocess.Popen(python_cmd, cwd=dir_name)
