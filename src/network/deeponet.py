# External Libs.
import torch
import torch.nn as nn
# Internal Libs.
from src.network.fcnn import FCNN as BaseNetwork
from src.network.resfcnn import ResidualFCNN as ResBaseNetwork
class DeepONet(nn.Module):
    def __init__(self, branchinfo, trunkinfo, channel_size, anainfo=[], name="DeepONet", is_resnet=False):
        super().__init__()
        self.m1_act_fn = branchinfo['act_fn']
        self.m2_act_fn = trunkinfo['act_fn']
        self.m1_input_size = branchinfo['input_size']
        self.m2_input_size = trunkinfo['input_size']
        self.m1_hidden_sizes = branchinfo['hidden_sizes']
        self.m2_hidden_sizes = trunkinfo['hidden_sizes']
        self.m1_output_size = branchinfo['output_size']
        self.m2_output_size = trunkinfo['output_size']
        self.model_channel_size = channel_size
        # model1: branch net
        # model2: trunk net
        if not is_resnet:
            self.branch_net = BaseNetwork(act_fn=self.m1_act_fn, input_size=self.m1_input_size, output_size=self.m1_output_size, hidden_sizes=self.m1_hidden_sizes)
            self.trunk_net = BaseNetwork(act_fn=self.m2_act_fn, input_size=self.m2_input_size, output_size=self.m2_output_size, hidden_sizes=self.m2_hidden_sizes)
        else:
            self.branch_net = ResBaseNetwork(act_fn=self.m1_act_fn, input_size=self.m1_input_size, output_size=self.m1_output_size, hidden_sizes=self.m1_hidden_sizes)
            self.trunk_net = ResBaseNetwork(act_fn=self.m2_act_fn, input_size=self.m2_input_size, output_size=self.m2_output_size, hidden_sizes=self.m2_hidden_sizes)
        self.config = {"branchinfo": branchinfo, "trunkinfo": trunkinfo, "channel_size": channel_size, "anainfo": anainfo, "name": name}

    def forward(self, x):
        x1 = x[:, 0 : self.m1_input_size]
        x2 = x[:, self.m1_input_size:]
        out1 = self.branch_net(x1)
        out2 = self.trunk_net(x2)
        out = self.batch_segmented_dot_product(out1, out2, self.model_channel_size)
        return out

    def batch_segmented_dot_product(self, branch_output, trunk_output, segment_sizes=None):
        if segment_sizes is None:
            segment_sizes = self.model_channel_size
        output_list = [branch_output] + [trunk_output]
        stacked_tensors = torch.stack(output_list)
        mul_tensors = torch.prod(stacked_tensors, dim=0)
        # Sum the mul_tensors at each row according to the segment_sizes
        results = torch.zeros(len(trunk_output), len(segment_sizes), device=trunk_output.device)
        start_idx = 0
        for j in range(len(segment_sizes)):
            length = segment_sizes[j]
            results[:, j] = mul_tensors[:, start_idx:start_idx + length].sum(dim=1)
            start_idx += length
        return results
    def branchnet_output(self, x_br):
        output = self.branch_net(x_br)
        return output

    def trunk_output(self, x_tr):
        output = self.trunk_net(x_tr)
        return output

    def forward_branch_fixed(self, branch_input, trunk_input, trunk_sample_num):
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)

        expanded_branch_output = torch.cat([branch_output[i].unsqueeze(0).repeat(trunk_sample_num[i], 1) for i in range(len(trunk_sample_num))],
                                   dim=0)
        out = self.batch_segmented_dot_product(expanded_branch_output, trunk_output, self.model_channel_size)
        return out

    def forward_branch_trunk_fixed(self, branch_input, trunk_input):
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)

        expanded_branch_output = torch.cat(
            [branch_output[i].unsqueeze(0).repeat(len(trunk_input), 1) for i in range(len(branch_input))],
            dim=0)
        # Repeat n trunk_output for each branch_output
        expanded_trunk_output = trunk_output.repeat(len(branch_input), 1)
        out = self.batch_segmented_dot_product(expanded_branch_output, expanded_trunk_output)
        return out


if __name__ == '__main__':
    branchinfo = {'act_fn': [nn.Tanh(), nn.Tanh(), nn.Tanh()], 'input_size': 3, 'output_size': 2, 'hidden_sizes': [32, 32, 32]}
    trunkinfo = {'act_fn': [nn.Tanh(), nn.Tanh(), nn.Tanh()], 'input_size': 1, 'output_size': 2, 'hidden_sizes': [32, 32, 32]}
    channel_size = [10]
    deeponet = DeepONet(branchinfo, trunkinfo, channel_size)
    branch_input = torch.randn(3, 3)
    trunk_input = torch.randn(2, 1).requires_grad_(True)
    out = deeponet.forward_branch_trunk_fixed(branch_input, trunk_input)
    print(out)
    from src.util.gradient import gradients
    ux = gradients(out, trunk_input)
    print(ux)

