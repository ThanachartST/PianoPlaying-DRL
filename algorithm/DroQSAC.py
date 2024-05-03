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
from core.ReplayBuffer import Transition
from common.EnvironmentSpec import EnvironmentSpec

    
@dataclass(frozen=True)
class DroQSACConfig:
    '''Configuration options for DroQSAC.'''

    num_Q: int = 2                                      # Number of ensemble of Q-functions 
    actor_lr: float = 3e-4                              # Learning rate of actor
    critic_lr: float = 3e-4                             # Learning rate of critc
    temp_lr: float = 3e-4                               # Learning rate of temperature (alpha)
    hidden_sizes: Sequence[int] = (256, 256, 256)       # Networks hidden sizes
    activation: str = "gelu"                            # Networks activation function
    num_min_Q: Optional[int] = 2                        # Number of ensemble for clipping Q-functions
    critic_dropout_rate: float = 0.0                    # Dropout rate in dropout layer on critic
    critic_layer_norm: bool = True                      # Flag: Apply layer norm in Q-networks
    rho: float = 0.005                                  # Hyperparameters for scaling in updating target Q-Networks params
    target_entropy: Optional[float] = None              # The target entropy using for calculating alpha loss
    init_temperature: float = 1.0                       # Initial values of temperture (alpha)
    backup_entropy: bool = True                         # FIXME: Not use
    q_target_mode: str = 'min'                          # FIXME: Not use, DroQ algorithm always use minimum Q-Networks
    auto_alpha: bool = True                             # Flag for the auto update temperature (alpha)
    device: str = 'cuda'                                # Processing device, default='cuda'

class DroQSACAgent(object):
    '''
    Naive SAC: num_Q = 2, num_min = 2
    REDQ: num_Q > 2, num_min = 2
    MaxMin: num_mins = num_Qs
    for above three variants, set q_target_mode to 'min' (default)
    Ensemble Average: set q_target_mode to 'ave'
    REM: set q_target_mode to 'rem'
    '''
    def __init__(self, 
                 spec: EnvironmentSpec, 
                 config: DroQSACConfig, 
                 gamma: float = 0.99):
        ''' 
        Initialize DroQ agent.

        Args:
            spec: Environment specification
            config: Agent configuration
            gamma: Discount factor

        '''
        # Action and observation dimensions
        act_dim = spec.action.shape[-1]
        obs_dim = spec.observation.shape[-1]

        # Intitializa temperater
        alpha = config.init_temperature

        # Set up policy network
        self.policy_net = TanhGaussianPolicy(obs_dim, act_dim, config.hidden_sizes).to(config.device)

        # Set up q networks
        self.q_net_list, self.q_target_net_list = [], []
        for q_i in range(config.num_Q):
            # Declare Q-networks
            new_q_net = MLP(obs_dim + act_dim, 1, config.hidden_sizes, target_drop_rate=config.critic_dropout_rate, layer_norm=config.critic_layer_norm).to(config.device)
            self.q_net_list.append(new_q_net)
            # Decalre target Q-networks, load the same parameters as main Q-netwokrs
            new_q_target_net = MLP(obs_dim + act_dim, 1, config.hidden_sizes, target_drop_rate=config.critic_dropout_rate, layer_norm=config.critic_layer_norm).to(config.device)
            new_q_target_net.load_state_dict(new_q_net.state_dict())
            self.q_target_net_list.append(new_q_target_net)

        # Set up policy optimizer
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=config.actor_lr)

        # Set up Q-function optimizer
        self.q_optimizer_list = []
        for q_i in range(config.num_Q):
            self.q_optimizer_list.append(optim.Adam(self.q_net_list[q_i].parameters(), lr=config.critic_lr))

        # Set up adaptive entropy (SAC adaptive)
        self.auto_alpha = config.auto_alpha
        if config.auto_alpha:
            self.target_entropy = config.target_entropy or -0.5 * act_dim

            self.log_alpha = torch.zeros(1, requires_grad=True, device=config.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=config.temp_lr)
            self.alpha = self.log_alpha.cpu().exp().item()
        else:
            self.alpha = alpha
            self.target_entropy, self.log_alpha, self.alpha_optim = None, None, None

        # Declare L2 norm function
        self.mse_criterion = nn.MSELoss()

        # Store hyperparameters
        self.gamma = gamma
        self.rho = config.rho
        self.num_min_Q = config.num_min_Q
        self.num_Q = config.num_Q
        self.q_target_mode = config.q_target_mode
        self.device = config.device

    def sample_actions(self, 
                       obs: Any):
        ''' 
        Sample actions from the given observation for training.

        Args:
            obs: Observation in array form with shape (obs_dim)

        Returns:
            action: Action in array form with shape (action_dim)
        
        '''
        # Given an observation, output a sampled action in numpy form
        with torch.no_grad():
            # Convert observation in array form to Tensor form
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
            # Sample action in Tensor form
            action_tensor = self.policy_net.forward(obs_tensor, deterministic=False,
                                            return_log_prob=False)[0]
            # Convert action in Tensor form to array form
            action = action_tensor.cpu().numpy().reshape(-1)

        return action

    def eval_actions(self, 
                     obs: Any):
        ''' 
        Sample deterministic actions from the given observation for evalution.

        Args:
            obs: Observation in array form with shape (obs_dim)

        Returns:
            action: Action in array form with shape (action_dim)
        
        '''
        # Given an observation, output a deterministic action in numpy form
        with torch.no_grad():
            # Convert observation in array form to Tensor form
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
            # Sample action in Tensor form
            action_tensor = self.policy_net.forward(obs_tensor, deterministic=True,
                                         return_log_prob=False)[0]
            # Convert action in Tensor form to array form
            action = action_tensor.cpu().numpy().reshape(-1)

        return action

    def get_droq_q_target_no_grad(self, 
                                  next_obs_tensor: Tensor, 
                                  reward_tensor: Tensor, 
                                  done: bool):
        ''' 
        Compute Q target.

        Args:
            next_obs_tensor: Observation in the next timestep in Tensor form with shape (batch size, obs_dim)
            reward_tensor: Reward in Tensor form with shape (batch size)
            done: bool whether an episode ends or not

        Returns:
            y_q: Q target value

        '''
        # compute REDQ Q target, depending on the agent's Q target mode
        # select Q networks that will be used
        # num_mins_to_use = get_probabilistic_num_min(self.num_min_Q)
        # sample_idxs = np.random.choice(self.num_Q, num_mins_to_use, replace=False)

        # Compute Q target value
        with torch.no_grad():
            if self.q_target_mode == 'min':
                '''Q target value is computed by min of a subset of Q values'''
                # Sample action from given observation in the next timestep
                a_tilda_next, _, _, log_prob_a_tilda_next, _, _ = self.policy_net.forward(next_obs_tensor)

                # Compute Q values in the next timestep
                q_prediction_next_list = []
                for sample_idx in range(self.num_Q):
                    q_prediction_next = self.q_target_net_list[sample_idx](torch.cat([next_obs_tensor, a_tilda_next], 1))
                    q_prediction_next_list.append(q_prediction_next)
                
                # Concat predicted Q-values.
                # Then, find minimum Q value from all Q values in the next timestep 
                q_prediction_next_cat = torch.cat(q_prediction_next_list, 1)
                min_q, min_indices = torch.min(q_prediction_next_cat, dim=1, keepdim=True)

                # Compute target Q-value
                next_q_with_log_prob = min_q - (self.alpha * log_prob_a_tilda_next)
                y_q = reward_tensor + self.gamma * (1 - done) * next_q_with_log_prob
            
        return y_q
    
    def soft_update_model1_with_model2(self, model1, model2, rho):
        ''' 
        Update Q target network.
        The update is model1 <- rho * model1 + (1 - rho) * model2

        Args:
            model1: pytorch model of Q target network
            model2: pytorch model of Q network
            rho: Hyperparameter for controling the update of Q target network

        '''
        for model1_param, model2_param in zip(model1.parameters(), model2.parameters()):
            model1_param.data.copy_(rho*model1_param.data + (1-rho)*model2_param.data)

    def update_actor(self, 
                     transitions: Transition):
        ''' 
        Update actor or policy network.

        Args:
            transitions: Transition object with keys ['state', 'action', 'reward', 'discount', 'next_state']

        Returns:
            Dict containing policy loss and entropy

        '''
        # Sample aprroximated action from the given observation
        a_tilda, _, _, log_prob_a_tilda, _, _ = self.policy_net.forward(transitions.state)

        # Compute Q-values from all Q networks
        q_a_tilda_list = []
        for sample_idx in range(self.num_Q):
            self.q_net_list[sample_idx].requires_grad_(False)
            q_a_tilda = self.q_net_list[sample_idx](torch.cat([transitions.state, a_tilda], 1))
            q_a_tilda_list.append(q_a_tilda)
        q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)

        # Find an average of Q values from all Q networks
        # return an array with shape (batch_size, )
        ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)

        # Compute policy loss, average along batch size
        policy_loss = ((self.alpha * log_prob_a_tilda) - ave_q).mean()

        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Resume requires_grad = True of Q networks
        # NOTE: we set requires_grad = False, above
        for sample_idx in range(self.num_Q):
            self.q_net_list[sample_idx].requires_grad_(True)

        return {"policy_loss": policy_loss, "batch_entropy": -log_prob_a_tilda}

    def update_critic(self, 
                      transitions: Transition):
        ''' 
        Update critic or Q networks.

        Args:
            transitions: Transition object with keys ['state', 'action', 'reward', 'discount', 'next_state']

        Returns:
            Dict containing Q losses from all Q networks

        '''
        # Compute Q target values from the given observation in the next timestpes and reward
        y_q = self.get_droq_q_target_no_grad(transitions.next_state, transitions.reward, done=0)

        # Compute Q values from all Q networks
        q_prediction_list = []
        for q_i in range(self.num_Q):
            q_prediction = self.q_net_list[q_i](torch.cat([transitions.state, transitions.action], 1))
            q_prediction_list.append(q_prediction)
        q_prediction_cat = torch.cat(q_prediction_list, dim=1)
        y_q = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q
        # Compute Q losses from all Q networks
        q_loss_all = self.mse_criterion(q_prediction_cat, y_q) * self.num_Q

        # Update Q networks
        for q_i in range(self.num_Q):
            self.q_optimizer_list[q_i].zero_grad()
        q_loss_all.backward()
        for q_i in range(self.num_Q):
            self.q_optimizer_list[q_i].step()
        
        return {"q_loss_all": q_loss_all}  

    def update_temperature(self, 
                           batch_entropy: Tensor):
        ''' 
        Update temperature.

        Args:
            batch_entropy: Entropy tensor with shape (batch_size,)

        Returns:
            Dict containing temperature loss (alpha_loss)

        '''
        if self.auto_alpha:
            # Compute temperature loss
            alpha_loss = (self.log_alpha * (batch_entropy - self.target_entropy).detach()).mean()
            # Update temperature
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.cpu().exp().item()
        else:
            alpha_loss = Tensor([0])

        return {"alpha_loss": alpha_loss}

    def update(self, 
               transitions: Transition):
        ''' 
        Update agent (actor, critic, temperature).

        Args:
            transitions: Transition object with keys ['state', 'action', 'reward', 'discount', 'next_state']

        Returns:
            new_agent: DroQSACAgent object
            Dict: containing temperature loss (alpha_loss)

        '''
        new_agent = self
        # Update critic
        critic_info = new_agent.update_critic(transitions)
        # Update actor
        actor_info = new_agent.update_actor(transitions)
        # Update temperature
        temp_info = new_agent.update_temperature(actor_info["batch_entropy"])

        # Update Q target networks
        for q_i in range(self.num_Q):
            self.soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i], self.rho)

        return new_agent, {**actor_info, **critic_info, **temp_info}       