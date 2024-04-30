# OPEN-SOURCE LIBRARY
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

# LOCAL LIBRARY
from core.Distribution import TanhGaussianPolicy
from core.Network import MLP


def get_probabilistic_num_min(num_mins):
    # allows the number of min to be a float
    floored_num_mins = np.floor(num_mins)
    if num_mins - floored_num_mins > 0.001:
        prob_for_higher_value = num_mins - floored_num_mins
        if np.random.uniform(0, 1) < prob_for_higher_value:
            return int(floored_num_mins+1)
        else:
            return int(floored_num_mins)
    else:
        return num_mins
    
@dataclass(frozen=True)
class DroQSACConfig:
    """Configuration options for SAC."""

    num_Q: int = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temp_lr: float = 3e-4
    hidden_sizes: Sequence[int] = (256, 256, 256)
    activation: str = "gelu"
    num_min_Q: Optional[int] = 2
    critic_dropout_rate: float = 0.0
    critic_layer_norm: bool = True
    rou: float = 0.005
    target_entropy: Optional[float] = None
    init_temperature: float = 1.0
    backup_entropy: bool = True
    q_target_mode: str = 'min'
    auto_alpha: bool = True
    device: str = 'cuda'

class DroQSACAgent(object):
    """
    Naive SAC: num_Q = 2, num_min = 2
    REDQ: num_Q > 2, num_min = 2
    MaxMin: num_mins = num_Qs
    for above three variants, set q_target_mode to 'min' (default)
    Ensemble Average: set q_target_mode to 'ave'
    REM: set q_target_mode to 'rem'
    """
    def __init__(self, spec, config, gamma=0.99):
        
        act_dim = spec.action.shape[-1]
        obs_dim = spec.observation.shape[-1]
        alpha = config.init_temperature

        # set up networks
        self.policy_net = TanhGaussianPolicy(obs_dim, act_dim, config.hidden_sizes).to(config.device)
        self.q_net_list, self.q_target_net_list = [], []
        for q_i in range(config.num_Q):
            # new_q_net = MLP(obs_dim + act_dim, 1, hidden_sizes).to(device)
            new_q_net = MLP(obs_dim + act_dim, 1, config.hidden_sizes, target_drop_rate=config.critic_dropout_rate, layer_norm=config.critic_layer_norm).to(config.device)
            self.q_net_list.append(new_q_net)
            # new_q_target_net = MLP(obs_dim + act_dim, 1, hidden_sizes).to(device)
            new_q_target_net = MLP(obs_dim + act_dim, 1, config.hidden_sizes, target_drop_rate=config.critic_dropout_rate, layer_norm=config.critic_layer_norm).to(config.device)
            new_q_target_net.load_state_dict(new_q_net.state_dict())
            self.q_target_net_list.append(new_q_target_net)
        # set up optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=config.actor_lr)
        self.q_optimizer_list = []
        for q_i in range(config.num_Q):
            self.q_optimizer_list.append(optim.Adam(self.q_net_list[q_i].parameters(), lr=config.critic_lr))
        # set up adaptive entropy (SAC adaptive)
        self.auto_alpha = config.auto_alpha
        if config.auto_alpha:
            self.target_entropy = config.target_entropy or -0.5 * act_dim

            self.log_alpha = torch.zeros(1, requires_grad=True, device=config.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=config.temp_lr)
            self.alpha = self.log_alpha.cpu().exp().item()
        else:
            self.alpha = alpha
            self.target_entropy, self.log_alpha, self.alpha_optim = None, None, None
        # set up replay buffer
        # self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        # set up other things
        self.mse_criterion = nn.MSELoss()

        # store other hyperparameters
        self.gamma = gamma
        self.rou = config.rou
        self.num_min_Q = config.num_min_Q
        self.num_Q = config.num_Q
        self.q_target_mode = config.q_target_mode
        self.device = config.device

    def sample_actions(self, obs):
        # given an observation, output a sampled action in numpy form
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
            action_tensor = self.policy_net.forward(obs_tensor, deterministic=False,
                                            return_log_prob=False)[0]
            action = action_tensor.cpu().numpy().reshape(-1)

        return action

    def eval_actions(self, obs):
        # given an observation, output a deterministic action in numpy form
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
            action_tensor = self.policy_net.forward(obs_tensor, deterministic=True,
                                         return_log_prob=False)[0]
            action = action_tensor.cpu().numpy().reshape(-1)
        return action

    def sample_actions_and_logprob_for_bias_evaluation(self, obs): #TODO modify the readme here
        # given an observation, output a sampled action in numpy form
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
            action_tensor, _, _, log_prob_a_tilda, _, _, = self.policy_net.forward(obs_tensor, deterministic=False,
                                         return_log_prob=True)
            action = action_tensor.cpu().numpy().reshape(-1)
        return action, log_prob_a_tilda

    def get_ave_q_prediction_for_bias_evaluation(self, obs_tensor, acts_tensor):
        # given obs_tensor and act_tensor, output Q prediction
        q_prediction_list = []
        for q_i in range(self.num_Q):
            q_prediction = self.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
            q_prediction_list.append(q_prediction)
        q_prediction_cat = torch.cat(q_prediction_list, dim=1)
        average_q_prediction = torch.mean(q_prediction_cat, dim=1)
        return average_q_prediction

    def get_droq_q_target_no_grad(self, obs_next_tensor, rews_tensor, done_tensor):
        # compute REDQ Q target, depending on the agent's Q target mode
        # allow min as a float:
        num_mins_to_use = get_probabilistic_num_min(self.num_min_Q)
        sample_idxs = np.random.choice(self.num_Q, num_mins_to_use, replace=False)
        with torch.no_grad():
            if self.q_target_mode == 'min':
                """Q target is min of a subset of Q values"""
                a_tilda_next, _, _, log_prob_a_tilda_next, _, _ = self.policy_net.forward(obs_next_tensor)
                q_prediction_next_list = []
                for sample_idx in sample_idxs:
                    q_prediction_next = self.q_target_net_list[sample_idx](torch.cat([obs_next_tensor, a_tilda_next], 1))
                    q_prediction_next_list.append(q_prediction_next)
                q_prediction_next_cat = torch.cat(q_prediction_next_list, 1)
                min_q, min_indices = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
                next_q_with_log_prob = min_q - self.alpha * log_prob_a_tilda_next
                y_q = rews_tensor + self.gamma * (1 - done_tensor) * next_q_with_log_prob
            # if self.q_target_mode == 'ave':
            #     """Q target is average of all Q values"""
            #     a_tilda_next, _, _, log_prob_a_tilda_next, _, _ = self.policy_net.forward(obs_next_tensor)
            #     q_prediction_next_list = []
            #     for q_i in range(self.num_Q):
            #         q_prediction_next = self.q_target_net_list[q_i](torch.cat([obs_next_tensor, a_tilda_next], 1))
            #         q_prediction_next_list.append(q_prediction_next)
            #     q_prediction_next_ave = torch.cat(q_prediction_next_list, 1).mean(dim=1).reshape(-1, 1)
            #     next_q_with_log_prob = q_prediction_next_ave - self.alpha * log_prob_a_tilda_next
            #     y_q = rews_tensor + self.gamma * (1 - done_tensor) * next_q_with_log_prob
            # if self.q_target_mode == 'rem':
            #     """Q target is random ensemble mixture of Q values"""
            #     a_tilda_next, _, _, log_prob_a_tilda_next, _, _ = self.policy_net.forward(obs_next_tensor)
            #     q_prediction_next_list = []
            #     for q_i in range(self.num_Q):
            #         q_prediction_next = self.q_target_net_list[q_i](torch.cat([obs_next_tensor, a_tilda_next], 1))
            #         q_prediction_next_list.append(q_prediction_next)
            #     # apply rem here
            #     q_prediction_next_cat = torch.cat(q_prediction_next_list, 1)
            #     rem_weight = Tensor(np.random.uniform(0, 1, q_prediction_next_cat.shape)).to(device=self.device)
            #     normalize_sum = rem_weight.sum(1).reshape(-1, 1).expand(-1, self.num_Q)
            #     rem_weight = rem_weight / normalize_sum
            #     q_prediction_next_rem = (q_prediction_next_cat * rem_weight).sum(dim=1).reshape(-1, 1)
            #     next_q_with_log_prob = q_prediction_next_rem - self.alpha * log_prob_a_tilda_next
            #     y_q = rews_tensor + self.gamma * (1 - done_tensor) * next_q_with_log_prob
        return y_q, sample_idxs
    
    def soft_update_model1_with_model2(self, model1, model2, rou):
        """
        used to polyak update a target network
        :param model1: a pytorch model
        :param model2: a pytorch model of the same class
        :param rou: the update is model1 <- rou*model1 + (1-rou)model2
        """
        for model1_param, model2_param in zip(model1.parameters(), model2.parameters()):
            model1_param.data.copy_(rou*model1_param.data + (1-rou)*model2_param.data)

    def update_actor(self, transitions):
        # get policy loss
        a_tilda, mean_a_tilda, log_std_a_tilda, log_prob_a_tilda, _, pretanh = self.policy_net.forward(transitions.state)
        q_a_tilda_list = []
        for sample_idx in range(self.num_Q):
            self.q_net_list[sample_idx].requires_grad_(False)
            q_a_tilda = self.q_net_list[sample_idx](torch.cat([transitions.state, a_tilda], 1))
            q_a_tilda_list.append(q_a_tilda)
        q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
        ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
        policy_loss = (self.alpha * log_prob_a_tilda - ave_q).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        for sample_idx in range(self.num_Q):
            self.q_net_list[sample_idx].requires_grad_(True)

        # update network
        self.policy_optimizer.step() 

        return {"policy_loss": policy_loss, "entropy": -log_prob_a_tilda.mean()}

    def update_critic(self, transitions):
        """Q loss"""
        y_q, sample_idxs = self.get_droq_q_target_no_grad(transitions.next_state, transitions.reward, done_tensor=0)
        q_prediction_list = []
        for q_i in range(self.num_Q):
            q_prediction = self.q_net_list[q_i](torch.cat([transitions.state, transitions.action], 1))
            q_prediction_list.append(q_prediction)
        q_prediction_cat = torch.cat(q_prediction_list, dim=1)
        y_q = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q
        q_loss_all = self.mse_criterion(q_prediction_cat, y_q) * self.num_Q
        for q_i in range(self.num_Q):
            self.q_optimizer_list[q_i].zero_grad()
        q_loss_all.backward()

        # update network
        for q_i in range(self.num_Q):
            self.q_optimizer_list[q_i].step()
        
        return {"q_loss_all": q_loss_all}  

    def update_temperature(self, entropy):
        # get alpha loss
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (entropy + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.cpu().exp().item()
        else:
            alpha_loss = Tensor([0])

        return {"alpha_loss": alpha_loss}

    def update(self, transitions):
        new_agent = self
        critic_info = new_agent.update_critic(transitions)
        actor_info = new_agent.update_actor(transitions)
        temp_info = new_agent.update_temperature(actor_info["entropy"])
        
        # polyak averaged Q target networks
        for q_i in range(self.num_Q):
            self.soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i], self.rou)

        return new_agent, {**actor_info, **critic_info, **temp_info}       