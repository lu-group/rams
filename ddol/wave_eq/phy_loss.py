import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import grf
from src.util.gradient import gradients

class PhyLoss():

    def __init__(self, branch_sam_num=1000, trunk_sam_num=10000, device=None):
        if device == None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.dtype = torch.float64
        self.recorded_interval = 80
        self.branch_sam_num = branch_sam_num
        self.trunk_sam_num = trunk_sam_num
        self.trunk_input = self.get_trunk_input(self.trunk_sam_num, self.device).to(self.dtype)
        self.trunk_input_ic = self.get_trunk_input_ic(device=self.device).to(self.dtype)
        self.trunk_input_bc = self.get_trunk_input_bc(device=self.device).to(self.dtype)
        self.branch_input = self.init_sample_branch_input(self.branch_sam_num, self.device)
        # self.branch_input = self.branch_input.to(self.dtype)

    def update_training_samples(self, net, kept_num, train_dataset, batch_size=32):
        net.to(self.dtype)
        net.eval()
        branch_input = self.get_branch_input(net, kept_num=kept_num, device=self.device)
        ic_list = branch_input.detach().cpu().numpy()
        results = []
        from solver import solver
        for i in range(len(ic_list)):
            grid_x, grid_t, U = solver(ic_list[i])
            results.append(U[::self.recorded_interval].reshape(-1))
        train_dataset.branch_input = torch.cat([train_dataset.branch_input,
                                                  torch.tensor(ic_list, dtype=torch.float32).to(self.device)], dim=0)
        added_label = torch.tensor(results, dtype=torch.float32).to(self.device)
        train_dataset.label = torch.cat([train_dataset.label, added_label], dim=0)
        added_label_norm = torch.norm(added_label, dim=1)
        train_dataset.label_norm = torch.cat([train_dataset.label_norm, added_label_norm], dim=0)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        net.to(torch.float32)
        net.train()
        return train_dataset, train_loader

    @staticmethod
    def get_celer(trunk_input, device='cpu'):
        # trunk_input: a tensor of the size n x 2 as [x, t]
        x = trunk_input[:, 0]
        # x <=0.7: c = 1.0; x > 0.7: c = 0.5
        c = torch.ones_like(x)
        c[x > 0.7] = 0.5
        return c.to(device)

    def init_sample_branch_input(self, num=1000, device='cpu'):
        x = np.linspace(0, 1, 201)
        branch_input, _ = grf.grf_1D(x, 0.3, num, L=None, is_torch=False, device=device)
        branch_input = np.array(branch_input)
        branch_input = branch_input * x * (1 - x)
        # branch_input_norm = np.linalg.norm(branch_input, axis=1) / 201 ** 0.5
        # branch_input = branch_input / branch_input_norm[:, None]
        branch_input = torch.tensor(branch_input).to(device)
        return branch_input

    def get_branch_input(self, net, kept_num=10, device='cpu'):
        branch_input = self.branch_input
        loss_phy = self.get_phy_loss(net, branch_input, self.trunk_input, device)
        loss_phy_norm = torch.norm(loss_phy, dim=1) / self.trunk_sam_num ** 0.5
        loss_bc = self.get_bc_loss(net, branch_input, self.trunk_input_bc, device)
        loss_bc_norm = torch.norm(loss_bc, dim=1) / self.trunk_sam_num ** 0.5
        loss_ic = self.get_ic_loss(net, branch_input, self.trunk_input_ic, device)
        loss_ic_norm = torch.norm(loss_ic, dim=1) / self.trunk_sam_num ** 0.5
        loss_norm = loss_phy_norm + loss_bc_norm + loss_ic_norm
        _, topk_indices = torch.topk(loss_norm, kept_num)
        branch_input = branch_input[topk_indices]
        return branch_input

    @staticmethod
    def get_trunk_input(num=10000, device='cpu'):
        # x ~ U(0, 1)
        x = torch.rand(num, 1).to(device)
        # t ~ U(0, 4)
        t = 4 * torch.rand(num, 1).to(device)
        return torch.cat([x, t], dim=1).to(device)

    @staticmethod
    def get_trunk_input_ic(device='cpu'):
        num = 201
        x = torch.linspace(0, 1, num).view(-1, 1).to(device)
        t = torch.zeros(num, 1).to(device)
        return torch.cat([x, t], dim=1).to(device)

    @staticmethod
    def get_trunk_input_bc(num=201, device='cpu'):
        x = torch.ones(num, 1)
        t = torch.linspace(0, 4, num).view(-1, 1)
        trunk_input_bc1 = torch.cat([x, t], dim=1)
        x = torch.zeros(num, 1)
        trunk_input_bc2 = torch.cat([x, t], dim=1)
        trunk_input_bc = torch.cat([trunk_input_bc1, trunk_input_bc2], dim=0)
        return trunk_input_bc.to(device)


    @staticmethod
    def get_phy_loss(net, branch_input, trunk_input, device='cpu'):
        c = PhyLoss.get_celer(trunk_input, device)
        # u_xx = []
        # u_tt = []
        # x, t = trunk_input[:, 0].detach(), trunk_input[:, 1].detach()
        # x.requires_grad = True
        # t.requires_grad = True
        # node = torch.stack((x, t), 1)
        # trunk_output = net.trunk_net(node)
        # for i in range(net.model_channel_size[0]):
        #     u = trunk_output[:, i].view(-1)
        #     print(i)
        #     u_xx.append(gradients(u, x, order=2))
        #     u_tt.append(gradients(u, t, order=2))
        # u_xx = torch.stack(u_xx, 0)
        # u_tt = torch.stack(u_tt, 0)
        # trunk_output_res = u_tt - c ** 2 * u_xx
        # branch_output = net.branch_net(branch_input)
        # res = torch.matmul(branch_output, trunk_output_res)
        # Compute the 2nd order derivative of u w.r.t. x and t using high-order FDM
        dx = 5e-3
        dt = 1e-3
        trunk_input_px1 = trunk_input + torch.tensor([dx, 0], dtype=torch.float64).to(device)
        trunk_input_px2 = trunk_input + torch.tensor([dx * 2, 0], dtype=torch.float64).to(device)
        trunk_input_mx1 = trunk_input - torch.tensor([dx, 0], dtype=torch.float64).to(device)
        trunk_input_mx2 = trunk_input - torch.tensor([dx * 2, 0], dtype=torch.float64).to(device)
        trunk_input_pt1 = trunk_input + torch.tensor([0, dt], dtype=torch.float64).to(device)
        trunk_input_pt2 = trunk_input + torch.tensor([0, dt * 2], dtype=torch.float64).to(device)
        trunk_input_mt1 = trunk_input - torch.tensor([0, dt], dtype=torch.float64).to(device)
        trunk_input_mt2 = trunk_input - torch.tensor([0, dt * 2], dtype=torch.float64).to(device)
        tpx1 = net.trunk_net(trunk_input_px1)
        tpx2 = net.trunk_net(trunk_input_px2)
        tmx1 = net.trunk_net(trunk_input_mx1)
        tmx2 = net.trunk_net(trunk_input_mx2)
        tpt1 = net.trunk_net(trunk_input_pt1)
        tpt2 = net.trunk_net(trunk_input_pt2)
        tmt1 = net.trunk_net(trunk_input_mt1)
        tmt2 = net.trunk_net(trunk_input_mt2)
        u_xx = (-tpx2 + 16 * tpx1 - 30 * net.trunk_net(trunk_input) + 16 * tmx1 - tmx2) / (12 * dx ** 2)
        u_tt = (-tpt2 + 16 * tpt1 - 30 * net.trunk_net(trunk_input) + 16 * tmt1 - tmt2) / (12 * dt ** 2)
        trunk_output_res = u_tt - c.view(-1, 1) ** 2 * u_xx
        branch_output = net.branch_net(branch_input)
        res = torch.matmul(branch_output, trunk_output_res.T)
        return res

    @staticmethod
    def get_bc_loss(net, branch_input, trunk_input, device='cpu'):
        trunk_output = net.trunk_net(trunk_input)
        branch_output = net.branch_net(branch_input)
        pred = torch.matmul(branch_output, trunk_output.T)
        res = pred - 0
        return res

    @staticmethod
    def get_ic_loss(net, branch_input, trunk_input, device='cpu'):
        trunk_output = net.trunk_net(trunk_input)
        branch_output = net.branch_net(branch_input)
        pred = torch.matmul(branch_output, trunk_output.T)
        res = pred - branch_input
        return res

if __name__ == '__main__':
    PhyLoss.get_trunk_input_bc()