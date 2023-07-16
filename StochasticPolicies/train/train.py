import os
import pickle
import torch
import numpy as np
import pandas as pd
import traci
import random

from StochasticPolicies.environments_framework.multi_signal import MultiSignal
from StochasticPolicies.PPO.PPO import IPPO
from .analysis import read_csv, read_xml
from StochasticPolicies.config.agent_config import agent_configs
from StochasticPolicies.config.map_config import map_configs
from StochasticPolicies.environments_framework import states, rewards

# train agents using the PPO algorithm
def train_PPO(
    run_name, map_name, map_dir, 
    episodes=1400, temperature=1.0, noise=0.0, layer_type=None, 
    update_interval=1024, minibatch_size=256, epochs=4, entropy_coef=0.001,  
    bbb_pi=0.5, bbb_sigma1=-0, bbb_sigma2=-6, 
    log_dir="./", reward_csv=None, 
    device="cpu", port=None, libsumo=False, 
    ):
    
    map_config = map_configs[map_name]
    agt_config = agent_configs["IPPO"]

    env_base = os.path.join(map_dir, "environments/")

    net_file = os.path.join(map_dir, map_config["net"])
    if map_config["route"] is None:
        route_file = map_config["route"]
    else:
        route_file = os.path.join(map_dir, map_config["route"])
    
    state_f = eval(agt_config["state"])
    reward_f = eval(agt_config["reward"])

    step_length = map_config["step_length"]
    yellow_length = map_config["yellow_length"]
    step_ratio = map_config["step_ratio"]
    start_time = map_config["start_time"]
    end_time = map_config["end_time"]
    max_distance = agt_config["max_distance"]
    lights = map_config["lights"]
    warmup = map_config["warmup"]

    csv_dir = log_dir + run_name + '-' + map_name + '-' + str(len(lights)) + '-' + state_f.__name__ + '-' + reward_f.__name__ + "/"

    env = MultiSignal(
        run_name=run_name, map_name=map_name,
        net=net_file, state_fn=state_f, reward_fn=reward_f,
        route=route_file, step_length=step_length, yellow_length=yellow_length, 
        step_ratio=step_ratio, end_time=end_time, max_distance=max_distance, 
        lights=lights, gui=False, log_dir=log_dir, libsumo=libsumo, 
        warmup=warmup, port=port)

    traffic_light_ids = env.all_ts_ids

    num_steps_eps = int((end_time - start_time) / step_length)

    agt_config['episodes'] = int(episodes * 0.8)    # schedulers decay over 80% of steps
    agt_config['steps'] = agt_config['episodes'] * num_steps_eps
    agt_config['log_dir'] = log_dir + env.connection_name + os.sep
    agt_config['num_lights'] = len(env.all_ts_ids)

    # Get agent id's, observation shapes, and action sizes from env
    obs_act = dict()
    for key in env.obs_shape:
        obs_act[key] = [env.obs_shape[key], len(env.phases[key]) if key in env.phases else None]
    
    
    model_param = {
        "temperature": temperature, "noise": noise, "layer_type": layer_type, 
        "bbb_pi": bbb_pi, "bbb_sigma1": bbb_sigma1, "bbb_sigma2": bbb_sigma2, 
        "device": device 
    }
    thread_number = 1
    agent = IPPO(agt_config, obs_act, map_name, thread_number, model_param, update_interval, minibatch_size, epochs, entropy_coef)
    
    play_steps = 0
    if layer_type == "noisy":
        agent.sample_noise()
    for _ in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            act = agent.act(obs)
            obs, rew, done, info = env.step(act)
            agent.observe(obs, rew, done, info)

            play_steps += 1
            if layer_type == "noisy" and play_steps % update_interval == 0:
                agent.sample_noise()

    env.close()

    if reward_csv is not None:
        episode_num = list()
        mean_reward = list()
        for i in range(episodes):
            episode_num.append(i+1)
            load_path = csv_dir + "metrics_" + str(i + 1) + ".csv"
            dataframe = pd.read_csv(load_path, header=None).dropna(axis=0)
            reward_sum = list()
            for j in range(len(dataframe.index)):
                reward_str = ",".join(list(dataframe.iloc[j, 1:1+len(traffic_light_ids)]))
                reward_sum.append(np.sum(list(eval(reward_str).values())))

            mean_reward.append(np.mean(reward_sum))

        analysis_data = {"episode": episode_num, "mean_reward": mean_reward}
        analysis_dataframe = pd.DataFrame(analysis_data)
        analysis_dataframe.to_csv(reward_csv, index=False)

    read_csv(log_dir)
    read_xml(log_dir, env_base)

    num = 1
    for model in agent.agents.values():
        filename = "agent" + str(num)
        path = os.path.join(log_dir, filename)
        model.save(path)
        num += 1