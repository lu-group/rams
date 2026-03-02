#############################################################################
# Fully-connected neural network (FCNN)
#############################################################################
import torch.nn as nn
import torch
import torch.nn.functional as F
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

class CNN(nn.Module):
    def __init__(self, input_size, activation1=F.tanh, activation2=F.tanh, outputsize=128):
        super(CNN, self).__init__()
        # Assuming activation function passed as an argument (default is ReLU)
        self.activation1 = activation1
        self.activation2 = activation2

        # Define the layers
        self.reshape = nn.Unflatten(1, (1, int(input_size ** 0.5), int(input_size ** 0.5)))  # Reshape input to (1, 20, 20)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2)  # First convolution layer
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)  # Second convolution layer
        self.conv3 = nn.Conv2d(32, 16, kernel_size=5, stride=2)  # Second convolution layer
        self.flatten = nn.Flatten()  # Flatten the output
        self.fc1 = nn.Linear(400, outputsize)  # First fully connected layer
        self.fc2 = nn.Linear(outputsize, outputsize)  # Second fully connected layer
        # self.fc3 = nn.Linear(outputsize, outputsize)

    def forward(self, x):
        x = self.reshape(x)  # Reshape input
        x = self.activation1(self.conv1(x))  # First conv layer with activation
        x = self.activation1(self.conv2(x))  # Second conv layer with activation
        x = self.activation1(self.conv3(x))
        x = self.flatten(x)  # Flatten the output
        x = self.activation2(self.fc1(x))  # First fully connected layer with activation
        x = self.activation2(self.fc2(x))  # Second fully connected layer (no activation here)
        # x = self.activation2(self.fc3(x))
        return x


class CNN2(nn.Module):
    def __init__(self, input_size, mlp, activation1=F.tanh):
        super(CNN2, self).__init__()
        # Assuming activation function passed as an argument (default is ReLU)
        self.activation1 = activation1
        self.reshape = nn.Unflatten(1, (1, int(input_size ** 0.5), int(input_size ** 0.5)))  # Reshape input to (1, 20, 20)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2)  # First convolution layer
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)  # Second convolution layer
        self.conv3 = nn.Conv2d(32, 16, kernel_size=5, stride=2)  # Second convolution layer
        self.flatten = nn.Flatten()  # Flatten the output
        self.mlp = mlp
        # self.fc3 = nn.Linear(outputsize, outputsize)

    def forward(self, x):
        x = self.reshape(x)  # Reshape input
        x = self.activation1(self.conv1(x))  # First conv layer with activation
        x = self.activation1(self.conv2(x))  # Second conv layer with activation
        x = self.activation1(self.conv3(x))
        x = self.flatten(x)  # Flatten the output
        x = self.mlp(x)
        return x
class MIONet(nn.Module):
    def __init__(self, branch_net1, branch_net2, trunk_net, channel_size=None):
        super(MIONet, self).__init__()
        self.branch_net1 = branch_net1
        self.branch_net2 = branch_net2
        self.trunk_net = trunk_net
        self.channel_size = channel_size

    def get_uv(self, branch_input1, branch_input2, trunk_input):
        branch_ourput1 = self.branch_net1(branch_input1)
        branch_output2 = self.branch_net2(branch_input2)
        trunk_output = self.trunk_net(trunk_input)
        branch_ourput = branch_ourput1 * branch_output2
        branch_output_u = branch_ourput[:, :self.channel_size]
        branch_output_v = branch_ourput[:, self.channel_size:]
        trunk_output_u = trunk_output[:, :self.channel_size]
        trunk_output_v = trunk_output[:, self.channel_size:]
        pred_u = torch.matmul(branch_output_u, trunk_output_u.transpose(0, 1))
        pred_v = torch.matmul(branch_output_v, trunk_output_v.transpose(0, 1))
        return pred_u, pred_v

    def forward(self, branch_input1, branch_input2, trunk_input):
        branch_ourput1 = self.branch_net1(branch_input1)
        branch_output2 = self.branch_net2(branch_input2)
        trunk_output = self.trunk_net(trunk_input)
        branch_ourput = branch_ourput1 * branch_output2
        output = torch.matmul(branch_ourput, trunk_output.transpose(0, 1))
        return output

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

class DeepONet(nn.Module):
    def __init__(self, branch_net, trunk_net, channel_size=None):
        super().__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.channel_size = channel_size


    def forward(self, branch_input, trunk_input):
        branch_ourput = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        output = torch.matmul(branch_ourput, trunk_output.transpose(0, 1))
        return output

if __name__ == '__main__':
    # layers = [2,100,100,100,100,100]  # Example layer sizes for a neural network
    # modified_model = ModifiedMLP(layers)
    # import torch
    # # Example input tensor
    # input_tensor = torch.randn(1, 2)
    # modified_output = modified_model(input_tensor)
    mesh_size = 64
    cnn = CNN(mesh_size * mesh_size)
    import torch
    # # Example input tensor
    input_tensor = torch.randn(1, mesh_size * mesh_size)
    cnn_output = cnn(input_tensor)
    # mlp = FCNN(input_size=3, output_size=2, hidden_sizes=[32, 32, 32], act_fn=[nn.Tanh(), nn.Tanh(), nn.Tanh()])
