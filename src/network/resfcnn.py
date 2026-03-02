#############################################################################
# Fully-connected neural network with residual blocks (ResFCNN)
#############################################################################
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, act_fn):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act_fn = act_fn
        # Apply skip connection if dimensions match
        self.use_skip_connection = in_features == out_features
        if not self.use_skip_connection and in_features != 0:
            # Linear transformation to match dimensions if needed
            self.dim_match = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        identity = x
        out = self.linear(x)
        if self.use_skip_connection:
            out += identity
        elif hasattr(self, 'dim_match'):
            identity = self.dim_match(identity)
            out += identity
        return self.act_fn(out)


class ResidualFCNN(nn.Module):
    def __init__(self, act_fn, input_size=3, output_size=2, hidden_sizes=[32, 32, 32], **kwargs):
        super(ResidualFCNN, self).__init__()
        layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            # Last layer without activation function
            if i == len(layer_sizes) - 2:
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            else:
                layers.append(ResidualBlock(layer_sizes[i], layer_sizes[i + 1], act_fn[i]))

        self.layers = nn.Sequential(*layers)

        # Store hyperparameters
        self.config = {
            "act_fn": act_fn.__class__.__name__,
            "input_size": input_size,
            "output_size": output_size,
            "hidden_sizes": hidden_sizes,
            "type": "ResFCNN",
            **kwargs
        }

    def forward(self, x):
        return self.layers(x)

if __name__ == '__main__':
    # Define the activation function
    act_fn = nn.Tanh()

    # Initialize the ResidualFCNN
    model = ResidualFCNN(act_fn=act_fn, input_size=10, output_size=2, hidden_sizes=[64, 64, 64])

    # Print the model structure
    print(model)