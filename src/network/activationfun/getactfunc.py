#############################################################################
# Get activation function from the input string
#############################################################################
import torch
import torch.nn as nn

class ActivationFunctionWrapper(nn.Module):
    def __init__(self, activation_function):
        super().__init__()
        self.activation_function = activation_function

    def forward(self, x):
        return self.activation_function(x)

def getactfn(act_fn_names):
    act_fn_by_name = {
        "Tanh": nn.Tanh(),
        "ReLU": nn.ReLU(),
        'Tanhshrink': nn.Tanhshrink(),
        "Softplus": nn.Softplus(),
        "Sigmoid": nn.Sigmoid(),
        "Sin": ActivationFunctionWrapper(torch.sin),
        "Cos": ActivationFunctionWrapper(torch.cos)
    }
    func_list = [act_fn_by_name[name] for name in act_fn_names]
    return func_list

