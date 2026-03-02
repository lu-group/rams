#############################################################################
# Fully-connected neural network (FCNN)
#############################################################################
import torch.nn as nn

class FCNN(nn.Module):
    def __init__(self, act_fn, input_size=3, output_size=2, hidden_sizes=[32,32,32], **kwargs):
        """
        Inputs:
            act_fn - Object of the activation function that should be used as non-linearity in the network.
            input_size - Size of the input
            output_size - Size of the output
            hidden_sizes - A list of integers specifying the hidden layer sizes in the NN
        """
        super().__init__()
        # Create the network based on the specified hidden sizes
        layers = []
        layer_sizes = [input_size] + hidden_sizes
        if type(act_fn) != type([]):
            tact_fn = act_fn
            act_fn = []
            for ii in range(len(layer_sizes) - 1):
                act_fn.append(tact_fn)
        for layer_index in range(1, len(layer_sizes)):
            layers += [nn.Linear(layer_sizes[layer_index-1], layer_sizes[layer_index]),
                       act_fn[layer_index - 1]]
        layers += [nn.Linear(layer_sizes[-1], output_size)]
        # nn.Sequential summarizes a list of modules into a single module, applying them in sequence
        self.layers = nn.Sequential(*layers)
        # We store all hyperparameters in a dictionary for saving and loading of the model
        act_fn_name = []
        for ii in act_fn:
            act_fn_name.append(ii._get_name())
        self.config = {"act_fn": act_fn_name,"input_size": input_size, "output_size": output_size,
                       "hidden_sizes": hidden_sizes, "type": "FCNN"}
        # Record the information in kwargs within the config
        for key, value in kwargs.items():
            self.config[key] = value

    def forward(self, x):
        out = self.layers(x)
        return out

import torch.nn.functional as F
class ModifiedMLP(nn.Module):
    def __init__(self, layers, activation=F.tanh):
        super(ModifiedMLP, self).__init__()
        self.activation = activation
        self.layers = nn.ModuleList()
        self.U1 = nn.Linear(layers[0], layers[1])
        self.U2 = nn.Linear(layers[0], layers[1])
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, x):
        U = self.activation(self.U1(x))
        V = self.activation(self.U2(x))
        for layer in self.layers[:-1]:
            outputs = self.activation(layer(x))
            x = outputs * U + (1 - outputs) * V
        x = self.layers[-1](x)
        return x
if __name__ == '__main__':
    layers = [2,100,100,100,100,100]  # Example layer sizes for a neural network
    modified_model = ModifiedMLP(layers)
    import torch
    # Example input tensor
    input_tensor = torch.randn(1, 2)
    modified_output = modified_model(input_tensor)
