# OPEN-SOURCE LIBRARY
import torch
import torch.nn as nn
from typing import Sequence, Any
from torch.nn import functional as F

# following SAC authors' and OpenAI implementation
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
ACTION_BOUND_EPSILON = 1e-6

def weights_init(linear_module: nn.Linear) -> None:
    '''
    Initial network weights by using xavier initialization.

    Args:
        linear_module: nn.Linear object
    '''
    if isinstance(linear_module, nn.Linear):
        torch.nn.init.xavier_uniform_(linear_module.weight, gain=1)
        # Initial with zero bias
        torch.nn.init.constant_(linear_module.bias, 0)


class MLP(nn.Module):
    def __init__(self,
                 input_size: Sequence[int] | Any,
                 output_size: Sequence[int] | Any,
                 hidden_sizes: Sequence[int],
                 hidden_activation = F.relu,
                 target_drop_rate: float = 0.0, 
                 layer_norm: bool = False) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation

        # ModuleList, the layers inside can be
        # detected by .parameters() call
        self.hidden_layers = nn.ModuleList()

        # Initialize each hidden layer
        in_size = input_size
        for _, next_size in enumerate(hidden_sizes):
            # The initial layer input shape equal to the input_size
            fc_layer = nn.Linear(in_size, next_size)
            # Update networs input size from hidden_sizes args
            in_size = next_size
            # Store linear layer into nn.ModuleList
            self.hidden_layers.append(fc_layer)

            # Add dropout and layer_norm after linear layer
            if target_drop_rate > 0.0:
                self.hidden_layers.append(nn.Dropout(p=target_drop_rate))  # dropout
            if layer_norm:
                self.hidden_layers.append(nn.LayerNorm(fc_layer.out_features))  # layer norm

        # The variable determine the number of layer 
        # before applying the activation function
        self.apply_activation_per = 1
        if target_drop_rate > 0.0:
            self.apply_activation_per += 1
        if layer_norm:
            self.apply_activation_per += 1

        # Init last fully connected layer with small weight and bias
        self.last_fc_layer = nn.Linear(in_size, output_size)
        self.apply(weights_init)

    def forward(self, input):
        h = input
        # Loop for hidden layers in nn.ModuleList
        for i, fc_layer in enumerate(self.hidden_layers):
            # Get the output from the current layer from input of the previous output
            h = fc_layer(h)
            # Apply the activation function, after N time of loop,
            # the maximum N is 3: ['fc', 'dropout', 'layer_norm']
            if ( (i + 1) % self.apply_activation_per) == 0:
                h = self.hidden_activation(h)
        output = self.last_fc_layer(h)
        return output
# Combined static and sequential data
# CSSD
class RnnMlp(nn.Module):
    def __init__(self,
                 seq_obs_dim: int,
                 static_obs_dim: int,
                 output_size: int,
                 rnn_hidden_size: int,
                 fc_hidden_sizes: Sequence[int],
                 hidden_activation = F.relu,
                 target_drop_rate: float = 0.0, 
                 layer_norm: bool = False) -> None:
        super().__init__()

        self.seq_obs_dim = seq_obs_dim
        self.n_features_static = static_obs_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation

        # ModuleList, the layers inside can be
        # detected by .parameters() call
        self.hidden_layers = nn.ModuleList()

        # Initialize each hidden layer
        # rnn for seq_input (goal)
        self.rnn = nn.RNN(seq_obs_dim, rnn_hidden_size, 1, batch_first=True, nonlinearity='relu')

        # NN that takes [rnn_output, static_input] as input
        fc_in_size = rnn_hidden_size + static_obs_dim
        for _, fc_next_size in enumerate(fc_hidden_sizes):
            # The initial layer input shape equal to the input_size
            fc_layer = nn.Linear(fc_in_size, fc_next_size)
            # Update networs input size from hidden_sizes args
            fc_in_size = fc_next_size
            # Store linear layer into nn.ModuleList
            self.hidden_layers.append(fc_layer)

            # Add dropout and layer_norm after linear layer
            if target_drop_rate > 0.0:
                self.hidden_layers.append(nn.Dropout(p=target_drop_rate))  # dropout
            if layer_norm:
                self.hidden_layers.append(nn.LayerNorm(fc_layer.out_features))  # layer norm

        # The variable determine the number of layer 
        # before applying the activation function
        self.apply_activation_per = 1
        if target_drop_rate > 0.0:
            self.apply_activation_per += 1
        if layer_norm:
            self.apply_activation_per += 1

        # Init last fully connected layer with small weight and bias
        self.last_fc_layer = nn.Linear(fc_in_size, output_size)
        self.apply(weights_init)

    def forward(self, seq_input, static_input):
        # Initialize hidden state with zeros
        # rnn_h_0 = torch.zeros(1, seq_input.size(0), self.rnn_hidden_size)

        # rnn with reverse of seq_input as input
        rnn_output, _ = self.rnn(torch.flip(seq_input, dims=[1]))

        # concat rnn_output and static_input
        # (batch_size, rnn_hidden_size + n_features_static)
        fc_h = torch.cat((rnn_output[:, -1, :], static_input), axis=1)

        # Loop for hidden layers in nn.ModuleList
        for i, fc_layer in enumerate(self.hidden_layers):
            # Get the output from the current layer from input of the previous output
            fc_h = fc_layer(fc_h)
            # Apply the activation function, after N time of loop,
            # the maximum N is 3: ['fc', 'dropout', 'layer_norm']
            if ((i + 1) % self.apply_activation_per) == 0:
                fc_h = self.hidden_activation(fc_h)
        output = self.last_fc_layer(fc_h)
        return output