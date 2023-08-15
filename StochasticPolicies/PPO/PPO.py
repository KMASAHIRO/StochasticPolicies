import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pfrl.initializers
from pfrl.agents import PPO

from StochasticPolicies.PPO.agent import IndependentAgent, Agent
from StochasticPolicies.layers.NoisyNet import NoisyLinear
from StochasticPolicies.layers.BBB import BayesianLinear

def lecun_init(layer, gain=1):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        pfrl.initializers.init_lecun_normal(layer.weight, gain)
        nn.init.zeros_(layer.bias)
    else:
        pfrl.initializers.init_lecun_normal(layer.weight_ih_l0, gain)
        pfrl.initializers.init_lecun_normal(layer.weight_hh_l0, gain)
        nn.init.zeros_(layer.bias_ih_l0)
        nn.init.zeros_(layer.bias_hh_l0)
    return layer

# Based on the Policy Function in RESCO
class DefaultModel(torch.nn.Module):
    def __init__(
        self, obs_space, act_space, num_batches, temperature=1.0, noise=0.0, 
        layer_type=None, bbb_pi=0.5, bbb_sigma1=-0, bbb_sigma2=-6, device="cpu"
        ):
        
        super().__init__()
        self.num_batches = num_batches
        self.temperature = temperature
        self.noise = noise
        self.layer_type = layer_type
        self.device = torch.device(device)
        num_hidden_units = 64

        def conv2d_size_out(size, kernel_size=2, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        h = conv2d_size_out(obs_space[1])
        w = conv2d_size_out(obs_space[2])

        self.conv = lecun_init(nn.Conv2d(obs_space[0], num_hidden_units, kernel_size=(2, 2)))
        self.flatten = nn.Flatten()
        self.linear1 = lecun_init(nn.Linear(h*w*num_hidden_units, num_hidden_units))
        self.linear2 = lecun_init(nn.Linear(num_hidden_units, num_hidden_units))
        if self.layer_type == "noisy":
            self.linear4_1 = NoisyLinear(num_hidden_units, act_space, device=self.device)
            self.linear4_2 = lecun_init(nn.Linear(num_hidden_units, 1))
        elif self.layer_type == "bbb":
            self.linear4_1 = BayesianLinear(num_hidden_units, act_space, device=self.device, pi=bbb_pi, sigma1=bbb_sigma1, sigma2=bbb_sigma2)
            self.linear4_2 = lecun_init(nn.Linear(num_hidden_units, 1))
        else:
            self.linear4_1 = lecun_init(nn.Linear(num_hidden_units, act_space), 1e-2)
            self.linear4_2 = lecun_init(nn.Linear(num_hidden_units, 1))
        self.relu = nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)

        if self.layer_type == "bbb":
            self.log_priors = list()
            self.log_variational_posteriors = list()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)

        if self.noise != 0.0 and self.training:
            x = x + torch.normal(torch.zeros(x.shape), torch.ones(x.shape)*self.noise).to(self.device)
        
        actions_outputs = self.linear4_1(x)
        actions_prob = self.softmax(actions_outputs/self.temperature)
        value = self.linear4_2(x)

        if self.layer_type == "bbb" and self.training:
            log_prior = self.linear4_1.log_prior
            log_variational_posterior = self.linear4_1.log_variational_posterior
                        
            self.log_priors.append(log_prior)
            self.log_variational_posteriors.append(log_variational_posterior)
        
        return torch.distributions.categorical.Categorical(actions_prob), value
    
    def return_bbb_info(self):
        return self.log_priors, self.log_variational_posteriors
    
    def reset_bbb_info(self):
        self.log_priors = list()
        self.log_variational_posteriors = list()
    
    def sample_noise(self):
        self.linear4_1.sample_noise()
    
    def remove_noise(self):
        self.linear4_1.remove_noise()
        
def _elementwise_clip(x, x_min, x_max):
    """Elementwise clipping

    Note: torch.clamp supports clipping to constant intervals
    """
    return torch.min(torch.max(x, x_min), x_max)

class STOCHASTIC_PPO(PPO):
    def _lossfun(
        self, entropy, vs_pred, log_probs, vs_pred_old, log_probs_old, advs, vs_teacher
        ):
        
        prob_ratio = torch.exp(log_probs - log_probs_old)

        loss_policy = -torch.mean(
            torch.min(
                prob_ratio * advs,
                torch.clamp(prob_ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs,
            ),
        )

        if self.clip_eps_vf is None:
            loss_value_func = F.mse_loss(vs_pred, vs_teacher)
        else:
            clipped_vs_pred = _elementwise_clip(
                vs_pred,
                vs_pred_old - self.clip_eps_vf,
                vs_pred_old + self.clip_eps_vf,
            )
            loss_value_func = torch.mean(
                torch.max(
                    F.mse_loss(vs_pred, vs_teacher, reduction="none"),
                    F.mse_loss(clipped_vs_pred, vs_teacher, reduction="none"),
                )
            )
        loss_entropy = -torch.mean(entropy)

        self.value_loss_record.append(float(loss_value_func))
        self.policy_loss_record.append(float(loss_policy))

        loss = (
            loss_policy
            + self.value_func_coef * loss_value_func
            + self.entropy_coef * loss_entropy
        )

        if hasattr(self.model, "layer_type"):
            if self.model.layer_type == "bbb":
                log_priors, log_variational_posteriors = self.model.return_bbb_info()
                log_prior = 0
                for i in range(len(log_priors)):
                    log_prior = log_prior + log_priors[i]
                log_prior = log_prior/len(log_priors)
                log_variational_posterior = 0
                for i in range(len(log_variational_posteriors)):
                    log_variational_posterior = log_variational_posterior + log_variational_posteriors[i]
                log_variational_posterior = log_variational_posterior/len(log_variational_posteriors)
                kl = (log_variational_posterior - log_prior)/self.model.num_batches

                loss = loss + kl

                self.model.reset_bbb_info()

        return loss
    
    def sample_noise(self):
        self.model.sample_noise()
    
    def remove_noise(self):
        self.model.remove_noise()

class IPPO(IndependentAgent):
    def __init__(self, config, obs_act, map_name, thread_number, model_param={}, update_interval=1024, minibatch_size=256, epochs=4, entropy_coef=0.001, load_path=[]):
        super().__init__(config, obs_act, map_name, thread_number)
        for key in obs_act:
            obs_space = obs_act[key][0]
            act_space = obs_act[key][1]
            self.agents[key] = PFRLPPOAgent(obs_space, act_space, model_param, update_interval, minibatch_size, epochs, entropy_coef)
        
        if load_path != []:
            i = 0
            for key in obs_act:
                self.agents[key].load(load_path[i])
                i += 1
    
    def sample_noise(self):
        for agent in self.agents.values():
            agent.sample_noise()
    
    def remove_noise(self):
        for agent in self.agents.values():
            agent.remove_noise()


class PFRLPPOAgent(Agent):
    def __init__(self, obs_space, act_space, model_param={}, update_interval=1024, minibatch_size=256, epochs=4, entropy_coef=0.001):
        super().__init__()

        model_param["obs_space"] = obs_space
        model_param["act_space"] = act_space
        model_param["num_batches"] = -(-update_interval//minibatch_size)
        self.model = DefaultModel(**model_param)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2.5e-4, eps=1e-5)
        
        if model_param.get("device"):
            self.device = torch.device(model_param["device"])

        self.agent = STOCHASTIC_PPO(
            self.model, self.optimizer, gpu=self.device.index,
            phi=lambda x: np.asarray(x, dtype=np.float32),
            clip_eps=0.1,
            clip_eps_vf=None,
            update_interval=update_interval,
            minibatch_size=minibatch_size,
            epochs=epochs,
            standardize_advantages=True,
            entropy_coef=entropy_coef,
            max_grad_norm=0.5)

    def act(self, observation):
        return self.agent.act(observation)

    def observe(self, observation, reward, done, info):
        self.agent.observe(observation, reward, done, False)

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path+'.pt')
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def sample_noise(self):
        self.agent.sample_noise()
    
    def remove_noise(self):
        self.agent.remove_noise()
