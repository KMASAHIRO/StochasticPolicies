import os
import numpy as np
import pandas as pd
import gym

from StochasticPolicies.REINFORCE.module import Agent

# 学習させる関数
def train_agent_gym(
    env_name, model_save_path=None, episode_per_learn=10, episodes=1400,  max_steps=500, 
    lr=0.01, decay_rate=0.01, temperature=1.0, noise=0.0, layer_type=None, 
    bbb_pi=0.5, bbb_sigma1=-0, bbb_sigma2=-6, 
    gamma=0.99, log_dir="./", learn_curve_csv=None, 
    device="cpu", gui=False):
    
    env = gym.make(env_name)
    num_states = 1
    for i in range(len(env.observation_space.shape)):
        num_states *= env.observation_space.shape[i]
    num_actions = env.action_space.n
    
    agent = Agent(
        num_states=num_states, num_actions=num_actions, 
        temperature=temperature, noise=noise, 
        layer_type=layer_type, lr=lr, decay_rate=decay_rate, 
        bbb_pi=bbb_pi, bbb_sigma1=bbb_sigma1, bbb_sigma2=bbb_sigma2,
        is_train=True, device=device)

    best_reward_sum = float("-inf")
    steps_list = list()
    loss_list = list()
    current_reward = list()
    
    if layer_type == "noisy":
        agent.sample_noise()

    for i in range(episodes):
        obs = env.reset()

        steps = 0
        episode_reward = list()
        for j in range(max_steps):
            chosen_actions = agent.act(obs)
            action = chosen_actions
            
            obs, reward, done, info = env.step(action)
            
            if gui:
                env.render()
            
            current_reward.append(reward)
            episode_reward.append(reward)

            steps += 1
            if done:
                steps_list.append(steps)
                R = 0
                for k in range(len(episode_reward)):
                    R = episode_reward[-(k+1)] + gamma*R
                    episode_reward[-(k+1)] = R
                agent.set_rewards(episode_reward)
                break

        if (i+1) % episode_per_learn == 0:
            loss = agent.train(return_loss=True)
            loss_list.append(loss)
            agent.reset_batch()

            if layer_type == "noisy":
                agent.sample_noise()

            current_reward_sum = np.sum(current_reward)
            if current_reward_sum > best_reward_sum:
                best_reward_sum = current_reward_sum
                current_reward = list()
                agent.save_model("best_" + model_save_path)
        
        print(env_name + ": episodes " + str(i + 1) + " ended", str(steps_list[-1]) + "steps")

    env.close()
    agent.reset_batch()

    if learn_curve_csv is not None:
        learn_num = -(-episodes // episode_per_learn)
        mean_steps = list()
        for i in range(learn_num):
            mean_steps.append(np.mean(steps_list[episode_per_learn * i:episode_per_learn * (i + 1)]))

        analysis_data = {"mean_steps": mean_steps, "loss": loss_list}
        analysis_dataframe = pd.DataFrame(analysis_data)
        analysis_dataframe.to_csv(os.path.join(log_dir, learn_curve_csv), index=False)

    agent.save_model(model_save_path)

