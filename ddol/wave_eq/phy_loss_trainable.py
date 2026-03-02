from phy_loss import PhyLoss as PhyLoss_Vanilla
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import grf
from src.util.gradient import gradients


class PhyLoss_Trainable(PhyLoss_Vanilla):
    def __init__(self, branch_sam_num=1000, trunk_sam_num=10000, device=None, lr=0.01, max_iter=500):
        super(PhyLoss_Trainable, self).__init__(branch_sam_num=branch_sam_num, trunk_sam_num=trunk_sam_num, device=device)
        self.lr = lr
        self.max_iter = max_iter

    def init_sample_branch_input(self, num=1000, device='cpu'):
        x = np.linspace(0, 1, 201)
        branch_input, _ = grf.grf_1D(x, 0.3, num, L=None, is_torch=False, device=device)
        branch_input = np.array(branch_input)
        return branch_input

    def get_branch_input(self, net, kept_num=10, device='cpu'):
        branch_input = self.branch_input
        x = np.linspace(0, 1, 201)
        tbranch_input = branch_input * x * (1 - x)
        tbranch_input = torch.tensor(tbranch_input).to(device)
        loss_phy = self.get_phy_loss(net, tbranch_input, self.trunk_input, device)
        loss_phy_norm = torch.norm(loss_phy, dim=1) / self.trunk_sam_num ** 0.5
        loss_bc = self.get_bc_loss(net, tbranch_input, self.trunk_input_bc, device)
        loss_bc_norm = torch.norm(loss_bc, dim=1) / self.trunk_sam_num ** 0.5
        loss_ic = self.get_ic_loss(net, tbranch_input, self.trunk_input_ic, device)
        loss_ic_norm = torch.norm(loss_ic, dim=1) / self.trunk_sam_num ** 0.5
        loss_norm = loss_phy_norm + loss_bc_norm + loss_ic_norm
        _, topk_indices = torch.topk(loss_norm, kept_num)
        branch_input = branch_input[topk_indices.detach().cpu().numpy()]

        branch_input = torch.tensor(branch_input, dtype=torch.float64).to(device)

        x = torch.tensor(x).to(device)
        x_diff = x[:, None] - x[None, :]
        sigma = 0.3
        kernel_matrix = torch.exp(-0.5 * (x_diff ** 2) / sigma ** 2).to(self.dtype)

        opt = torch.optim.Adam(params=[branch_input], lr=self.lr)
        net.eval()
        for _ in range(self.max_iter):
            if _ != 0:
                tsam = (kernel_matrix @ branch_input.T / kernel_matrix.sum(dim=1, keepdim=True)).T
            else:
                tsam = branch_input
            tsam = tsam * x * (1 - x)
            opt.zero_grad()
            loss_phy = torch.sum(self.get_phy_loss(net, tsam, self.trunk_input, device) ** 2)
            loss_bc = torch.sum(self.get_bc_loss(net, tsam, self.trunk_input_bc, device) ** 2)
            loss_ic = torch.sum(self.get_ic_loss(net, tsam, self.trunk_input_ic, device) ** 2)
            loss = -1 * (loss_phy + loss_bc + loss_ic)
            loss.backward()
            opt.step()
            # Project the samples back to the original space with the function norm = 1
        # new_sam.requires_grad_(False)
        branch_input = (kernel_matrix @ branch_input.T / kernel_matrix.sum(dim=1, keepdim=True)).T
        branch_input = branch_input * x * (1 - x)
        net.train()
        branch_input = branch_input.detach()
        return branch_input
