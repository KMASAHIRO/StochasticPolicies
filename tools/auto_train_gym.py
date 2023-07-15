import os
import configparser
import subprocess

if __name__ == "__main__":
    config_gym = configparser.ConfigParser()
    config_gym.read("config_gym.ini", encoding='utf-8')

    train_path = config_gym.get("DEFAULT", "do_train_path")
    env_name = config_gym.get("DEFAULT", "env_name")
    
    sections = config_gym.sections()
    experiments = dict()
    for sec in sections:
        if sec != "DEFAULT":
            experiments[sec] = dict(config_gym.items(sec))
            experiments[sec].pop("do_train_path")
            if "learn_curve_csv" not in experiments[sec].keys():
                experiments[sec]["learn_curve_csv"] = env_name + "_" + sec + "_learncurve.csv"
            if "model_save_path" not in experiments[sec].keys():
                experiments[sec]["model_save_path"] = env_name + "_" + sec + "_policy-function.pth"
            if "log_dir" not in experiments[sec].keys():
                experiments[sec]["log_dir"] = "./"


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
