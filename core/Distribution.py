# OPEN-SOURCE LIBRARY
import torch
import torch.nn as nn
from typing import Sequence, Tuple
from torch.nn import functional as F
from torch.distributions import Normal

# LOCAL LIBRARY
from core.Network import MLP, weights_init

# following SAC authors' and OpenAI implementation
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
ACTION_BOUND_EPSILON = 1e-6

class TanhGaussianPolicy(MLP):
    '''
    A Gaussian policy network with Tanh to enforce action limits.
    '''

    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 hidden_sizes: Sequence[int],
                 hidden_activation = F.relu,
                 action_limit: float = 1.0) -> None:
        '''
        '''
        super().__init__(input_size = obs_dim,
                         output_size = action_dim,
                         hidden_sizes = hidden_sizes,
                         hidden_activation = hidden_activation)
        
        last_hidden_size = obs_dim
        
        if len(hidden_sizes) > 0:
            last_hidden_size = hidden_sizes[-1]
        
        # The layer that gives log_std, initalize this layer with small weight and bias
        # Output shape (action_dim, )
        self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
        
        # The action limit: for example, humanoid has an action limit of -0.4 to 0.4
        self.action_limit = action_limit

        # Apply function into nn.Module
        self.apply(weights_init)

    def forward(self,
                obs: torch.Tensor,
                deterministic: bool = False,
                return_log_prob: bool = True) -> Tuple:
        '''
        Compute the policy networks from the observation given.

        Args:
            obs: the observation tensor with shape ( batch_size, obs_dim )
            deterministic: Deterministic flag, default = False
                The deterministic flag control the action behavior,
                sample from the probability if True. Otherwise, using mean instead.
            return_log_prob: Flag for controlling the output log_prob variable

        Returns:
            Tuple contains with [action, mean, log_std, log_prob, std, pre_tanh_value]

        '''
        h = obs
        # Loop for all module in nn.ModuleList object
        for fc_layer in self.hidden_layers:
            
            # NOTE: DroQ policy network not use the dropout and layer norm
            # The calculation will not the same as MLP class
            h = self.hidden_activation(fc_layer(h))

        # Get mean from the last fc layer from MLP object (Parent class)
        mean = self.last_fc_layer(h)

        # Get log_std, from the last fc layer from TanhGaussianPolicy object
        # by using the same output from MLP
        log_std = self.last_fc_log_std(h)

        # Clamp the SD, with the DEFAULT MIN & MAX value
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        
        # Take exponential function, convert log-scale into normal scale
        std = torch.exp(log_std)

        # Declare normal distribution object
        try:
            normal_dist = Normal(mean, std)
        except:
            print( f'\nObservations: {obs}' )
            print( f'Last hidden output: { h }' )
            print( f'Mean: { mean }' )
            print( f'self.last_fc_layer weight: { self.last_fc_layer.state_dict() }')
            print( f'Std: {std}\n' )
            raise ValueError()

        # Deterministic action, Using on the evaluation taks
        if deterministic:
            pre_tanh_value = mean
            action = torch.tanh(mean)

        # Stochastic action, Using on the training tasks
        # sample action from the distribution probability
        else:
            pre_tanh_value = normal_dist.rsample()
            # print( f'pre_tanh_value shape : { pre_tanh_value.shape }' )
            action = torch.tanh(pre_tanh_value)
            # print( f'action.shape: { action.shape }' )

        # Calculate the log probability of action, if return flag is True
        if return_log_prob:
            #   Return tensor shape (batch_size, action_dim)
            log_prob = normal_dist.log_prob(pre_tanh_value)
            log_prob -= torch.log(1 - action.pow(2) + ACTION_BOUND_EPSILON)
            # Sum along action axis,
            log_prob = log_prob.sum(1, keepdim=True)
        
        # If return flag is False, log_prob will be None
        else:
            log_prob = None

        return ( ( action * self.action_limit ), mean, log_std, log_prob, std, pre_tanh_value )