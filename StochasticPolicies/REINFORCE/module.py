import torch
import numpy as np
from StochasticPolicies.layers.NoisyNet import NoisyLinear
from StochasticPolicies.layers.BBB import BayesianLinear

# Policy Function
class PolicyFunction(torch.nn.Module):
    def __init__(
        self, num_states, num_actions, temperature, noise, layer_type, 
        bbb_pi, bbb_sigma1, bbb_sigma2, device):
        
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.temperature = temperature
        self.noise = noise
        self.layer_type = layer_type
        self.device = torch.device(device)
        num_hidden_units = 32

        self.fc_first = torch.nn.Linear(self.num_states, num_hidden_units)
        self.fc_mid_layer = torch.nn.Linear(num_hidden_units, num_hidden_units)
        
        if self.layer_type == "noisy":
            self.fc_last_layer = NoisyLinear(num_hidden_units, self.num_actions, device=self.device)
        elif self.layer_type == "bbb":
            self.fc_last_layer = BayesianLinear(num_hidden_units, self.num_actions, device=self.device, pi=bbb_pi, sigma1=bbb_sigma1, sigma2=bbb_sigma2)
        else:
            self.fc_last_layer = torch.nn.Linear(num_hidden_units, self.num_actions)
        
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, inputs):
        x = self.fc_first(inputs)
        x = self.relu(x)
        x = self.fc_mid_layer(x)
        x = self.relu(x)
        
        if self.training:
            x = x + torch.normal(torch.zeros(x.shape[-1]), torch.ones(x.shape[-1])*self.noise).to(self.device)

        last_x = self.fc_last_layer(x)
        last_outputs = self.softmax(last_x/self.temperature)
        
        if self.layer_type == "bbb" and self.training:
            log_prior = self.fc_last_layer.log_prior
            log_variational_posterior = self.fc_last_layer.log_variational_posterior
            
            return last_outputs, log_prior, log_variational_posterior
        else:
            return last_outputs
    
    def sample_noise(self):
        self.fc_last_layer.sample_noise()
    
    def remove_noise(self):
        self.fc_last_layer.remove_noise()

# Loss Function
class PolicyGradientLossWithREINFORCE(torch.nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, actions_prob_history, rewards_history, beta=0.25, beta_loss_history=None):
        ave_rewards = np.mean(rewards_history)
        loss = 0
        for i in range(len(actions_prob_history)):
            chosen_action_prob = actions_prob_history[i]
            # Negative of the function to maximize
            loss = loss - torch.log(chosen_action_prob) * (rewards_history[i] - ave_rewards)
            
        loss = loss / len(actions_prob_history)

        return loss

class Agent():
    def __init__(
        self, num_states, num_actions, temperature, noise, layer_type, 
        lr, decay_rate, bbb_pi, bbb_sigma1, bbb_sigma2, 
        is_train, device
        ):
        
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr
        self.layer_type = layer_type
        self.is_train = is_train
        self.device = torch.device(device)
        
        self.policy_function = PolicyFunction(
            self.num_states, self.num_actions, 
            temperature, noise, layer_type, 
            bbb_pi, bbb_sigma1, bbb_sigma2, device)
        self.policy_function.to(self.device)

        if self.is_train:
            self.actions_prob_history = list()
            self.rewards_history = list()
            if self.layer_type == "bbb":
                self.log_priors = list()
                self.log_variational_posteriors = list()
            self.policy_function.train()
            self.loss_f = PolicyGradientLossWithREINFORCE()
            param_optimizer = list(self.policy_function.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': decay_rate},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            self.optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=self.lr)
        else:
            self.policy_function.eval()

    def act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)

        if self.is_train:
            if self.layer_type == "bbb":
                actions_prob, log_prior, log_variational_posterior = self.policy_function(obs)
            else:
                actions_prob = self.policy_function(obs)
        else:
            with torch.no_grad():
                actions_prob = self.policy_function(obs)
        
        prob_numpy = actions_prob.to("cpu").detach().numpy()
        chosen_actions = np.random.choice(
                a=np.arange(len(prob_numpy)), 
                size=1, 
                replace=True, 
                p=prob_numpy
                )[0]

        if self.is_train:
            self.actions_prob_history.append(actions_prob[chosen_actions])
            if self.layer_type == "bbb":
                self.log_priors.append(log_prior)
                self.log_variational_posteriors.append(log_variational_posterior)
        
        return chosen_actions

    def train(self, return_loss=False):
        if self.layer_type == "bbb":
            log_prior = 0
            for i in range(len(self.log_priors)):
                log_prior = log_prior + self.log_priors[i]
            log_prior = log_prior/len(self.log_priors)
            log_variational_posterior = 0
            for i in range(len(self.log_variational_posteriors)):
                log_variational_posterior = log_variational_posterior + self.log_variational_posteriors[i]
            log_variational_posterior = log_variational_posterior/len(self.log_variational_posteriors)
            kl = (log_variational_posterior - log_prior)/1
            
            loss = self.loss_f(self.actions_prob_history, self.rewards_history) + kl
        else:
            loss = self.loss_f(self.actions_prob_history, self.rewards_history)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if return_loss:
            return loss.to("cpu").detach().numpy()

    def set_rewards(self, reward):
        if self.is_train:
            self.rewards_history.extend(reward)

    def reset_batch(self):
        if self.is_train:
            self.actions_prob_history = list()
            self.rewards_history = list()
            if self.layer_type == "bbb":
                self.log_priors = list()
                self.log_variational_posteriors = list()

    def save_model(self, path):
        self.policy_function.to("cpu")
        torch.save(self.policy_function.state_dict(), path)
        
        self.policy_function.to(self.device)
    
    def sample_noise(self):
        self.policy_function.sample_noise()
    
    def remove_noise(self):
        self.policy_function.remove_noise()
