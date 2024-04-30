import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Distribution, Normal

from typing import NamedTuple, Optional
import numpy as np
import dm_env

# following SAC authors' and OpenAI implementation
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
ACTION_BOUND_EPSILON = 1E-6
# these numbers are from the MBPO paper
mbpo_target_entropy_dict = {'Hopper-v2':-1, 'HalfCheetah-v2':-3, 'Walker2d-v2':-3, 'Ant-v2':-4, 'Humanoid-v2':-2}
mbpo_epoches = {'Hopper-v2':125, 'Walker2d-v2':300, 'Ant-v2':300, 'HalfCheetah-v2':400, 'Humanoid-v2':300}

def weights_init_(m):
    # weight init helper function
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Transition(NamedTuple):
    state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    discount: np.ndarray
    next_state: np.ndarray

class TransitionTensor(NamedTuple):
    state: Tensor
    action: Tensor
    reward: Tensor
    discount: Tensor
    next_state: Tensor


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_size: int,
        batch_size: int,
    ) -> None:
        self._max_size = max_size
        self._batch_size = batch_size

        # Storage.
        self._states = np.zeros((max_size, state_dim), dtype=np.float32)
        self._actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self._next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self._rewards = np.zeros((max_size), dtype=np.float32)
        self._discounts = np.zeros((max_size), dtype=np.float32)

        self._ptr: int = 0
        self._size: int = 0
        self._prev: Optional[dm_env.TimeStep] = None
        self._action: Optional[np.ndarray] = None
        self._latest: Optional[dm_env.TimeStep] = None

    def insert(
        self,
        timestep: dm_env.TimeStep,
        action: Optional[np.ndarray],
    ) -> None:
        self._prev = self._latest
        self._action = action
        self._latest = timestep

        if action is not None:
            self._states[self._ptr] = self._prev.observation  # type: ignore
            self._actions[self._ptr] = action
            self._next_states[self._ptr] = self._latest.observation
            self._rewards[self._ptr] = self._latest.reward
            self._discounts[self._ptr] = self._latest.discount

            self._ptr = (self._ptr + 1) % self._max_size
            self._size = min(self._size + 1, self._max_size)

    def sample(self, device):
        self._ind = np.random.randint(0, self._size, size=self._batch_size)
        # return Transition(
        #     state=self._states[self._ind],
        #     action=self._actions[self._ind],
        #     reward=self._rewards[self._ind],
        #     discount=self._discounts[self._ind],
        #     next_state=self._next_states[self._ind],
        # )
        batch = Transition(
            state=self._states[self._ind],
            action=self._actions[self._ind],
            reward=self._rewards[self._ind],
            discount=self._discounts[self._ind],
            next_state=self._next_states[self._ind],
        )

        return TransitionTensor(
            state=Tensor(batch.state).to(device),
            action=Tensor(batch.action).to(device),
            reward=Tensor(batch.reward).unsqueeze(1).to(device),
            discount=Tensor(batch.discount).unsqueeze(1).to(device),
            next_state=Tensor(batch.next_state).to(device)
        )

    def is_ready(self) -> bool:
        return self._batch_size <= len(self)

    def __len__(self) -> int:
        return self._size


class Mlp(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_sizes,
            hidden_activation=F.relu,
            #
            target_drop_rate = 0.0, layer_norm = False

    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        ## here we use ModuleList so that the layers in it can be
        ## detected by .parameters() call
        self.hidden_layers = nn.ModuleList()
        in_size = input_size

        ## initialize each hidden layer
        for i, next_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(in_size, next_size)
            in_size = next_size
            self.hidden_layers.append(fc_layer)

            # added 20211206
            if target_drop_rate > 0.0:
                self.hidden_layers.append(nn.Dropout(p=target_drop_rate))  # dropout
            if layer_norm:
                self.hidden_layers.append(nn.LayerNorm(fc_layer.out_features))  # layer norm

        # added to fix bug 20211207
        self.apply_activation_per = 1
        if target_drop_rate > 0.0:
            self.apply_activation_per += 1
        if layer_norm:
            self.apply_activation_per += 1

        ## init last fully connected layer with small weight and bias
        self.last_fc_layer = nn.Linear(in_size, output_size)
        self.apply(weights_init_)

    def forward(self, input):
        h = input
        for i, fc_layer in enumerate(self.hidden_layers):
            h = fc_layer(h)
            #h = self.hidden_activation(h)
            if ( (i + 1) % self.apply_activation_per) == 0:
                h = self.hidden_activation(h)
        output = self.last_fc_layer(h)
        return output

class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)
    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def log_prob(self, value, pre_tanh_value=None):
        """
        return the log probability of a value
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        # use arctanh formula to compute arctanh(value)
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1+value) / (1-value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - \
               torch.log(1 - value * value + self.epsilon)

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.
        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        Implement: tanh(mu + sigma * eksee)
        with eksee~N(0,1)
        z here is mu+sigma+eksee
        """
        z = (
            self.normal_mean +
            self.normal_std *
            Normal( ## this part is eksee~N(0,1)
                torch.zeros(self.normal_mean.size()),
                torch.ones(self.normal_std.size())
            ).sample()
        )
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

class TanhGaussianPolicy(Mlp):
    """
    A Gaussian policy network with Tanh to enforce action limits
    """
    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_sizes,
            hidden_activation=F.relu,
            action_limit=1.0
    ):
        super().__init__(
            input_size=obs_dim,
            output_size=action_dim,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
        )
        last_hidden_size = obs_dim
        if len(hidden_sizes) > 0:
            last_hidden_size = hidden_sizes[-1]
        ## this is the layer that gives log_std, init this layer with small weight and bias
        self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
        ## action limit: for example, humanoid has an action limit of -0.4 to 0.4
        self.action_limit = action_limit
        self.apply(weights_init_)

    def forward(
            self,
            obs,
            deterministic=False,
            return_log_prob=True,
    ):
        """
        :param obs: Observation
        :param reparameterize: if True, use the reparameterization trick
        :param deterministic: If True, take determinisitc (test) action
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for fc_layer in self.hidden_layers:
            h = self.hidden_activation(fc_layer(h))
        mean = self.last_fc_layer(h)

        log_std = self.last_fc_log_std(h)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        normal = Normal(mean, std)

        if deterministic:
            pre_tanh_value = mean
            action = torch.tanh(mean)
        else:
            pre_tanh_value = normal.rsample()
            action = torch.tanh(pre_tanh_value)

        if return_log_prob:
            log_prob = normal.log_prob(pre_tanh_value)
            log_prob -= torch.log(1 - action.pow(2) + ACTION_BOUND_EPSILON)
            log_prob = log_prob.sum(1, keepdim=True)
        else:
            log_prob = None

        return (
            action * self.action_limit, mean, log_std, log_prob, std, pre_tanh_value,
        )

def soft_update_model1_with_model2(model1, model2, rou):
    """
    used to polyak update a target network
    :param model1: a pytorch model
    :param model2: a pytorch model of the same class
    :param rou: the update is model1 <- rou*model1 + (1-rou)model2
    """
    for model1_param, model2_param in zip(model1.parameters(), model2.parameters()):
        model1_param.data.copy_(rou*model1_param.data + (1-rou)*model2_param.data)

def test_agent(agent, test_env, max_ep_len, logger, n_eval=1):
    """
    This will test the agent's performance by running <n_eval> episodes
    During the runs, the agent should only take deterministic action
    This function assumes the agent has a <get_test_action()> function
    :param agent: agent instance
    :param test_env: the environment used for testing
    :param max_ep_len: max length of an episode
    :param logger: logger to store info in
    :param n_eval: number of episodes to run the agent
    :return: test return for each episode as a numpy array
    """
    ep_return_list = np.zeros(n_eval)
    for j in range(n_eval):
        o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
        while not (d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time
            a = agent.get_test_action(o)
            o, r, d, _ = test_env.step(a)
            ep_ret += r
            ep_len += 1
        ep_return_list[j] = ep_ret
        if logger is not None:
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
    return ep_return_list
