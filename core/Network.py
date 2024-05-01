# OPEN-SOURCE LIBRARY
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Sequence,Any

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