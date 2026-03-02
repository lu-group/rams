import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

class PhyLoss():
    mu = 0.02
    proj_l = 0.3

    def __init__(self, branch_sam_num=1000, trunk_sam_num=10000, device=None, max_iter=1000, lr=1e-2):
        if device == None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.dtype = torch.float64
        self.branch_sam_num = branch_sam_num
        self.trunk_sam_num = trunk_sam_num
        self.trunk_input = self.get_trunk_input(self.trunk_sam_num, self.device).to(self.dtype)
        self.trunk_input_ic = self.get_trunk_input_ic(device=self.device).to(self.dtype)
        self.trunk_input_bc = self.get_trunk_input_bc(device=self.device).to(self.dtype)
        self.branch_input = self.init_sample_branch_input(self.branch_sam_num)
        self.max_iter = max_iter
        self.lr = lr

    def init_sample_branch_input(self, num=1000):
        nx = 64
        ny = 64
        x = np.linspace(-1, 1, nx)  # Coordinate Along X direction
        y = np.linspace(-1, 1, ny)  # Coordinate Along Y direction
        X, Y = np.meshgrid(x, y)
        orignal_node = np.stack((X.flatten(), Y.flatten()), axis=1)
        from grf import grf_2D
        u0_list, _ = grf_2D(orignal_node, PhyLoss.proj_l, num, std=1, L=None, is_torch=False, device=None)
        return u0_list

    def get_branch_input(self, net, kept_num=10, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        branch_input = self.branch_input
        x = np.linspace(-1, 1, 64)  # Coordinate Along X direction
        y = np.linspace(-1, 1, 64)  # Coordinate Along Y direction
        X, Y = np.meshgrid(x, y)
        node = np.stack((X.flatten(), Y.flatten()), axis=1)
        tbranch_input = branch_input * (1 - X.flatten() ** 2) * (1 - Y.flatten() ** 2) / 2
        tbranch_input = torch.tensor(tbranch_input).to(device)
        loss_phy = self.get_phy_loss(net, tbranch_input, self.trunk_input, device)
        loss_phy_mean = torch.mean(loss_phy ** 2, dim=1)
        loss_bc = self.get_bc_loss(net, tbranch_input, self.trunk_input_bc, device)
        loss_bc_mean = torch.mean(loss_bc ** 2, dim=1)
        loss_ic = self.get_ic_loss(net, tbranch_input, self.trunk_input_ic, device)
        loss_ic_mean = torch.mean(loss_ic ** 2, dim=1)
        loss_mean = loss_phy_mean + loss_bc_mean + loss_ic_mean
        _, topk_indices = torch.topk(loss_mean, kept_num)
        branch_input = branch_input[topk_indices.detach().cpu().numpy()]

        def gaussian_kernel(x, y, l):
            """Generate a Gaussian kernel matrix with correlation length l."""
            # Efficiently calculate the squared exponential kernel
            dx = x[:, np.newaxis] - x[np.newaxis, :]
            dy = y[:, np.newaxis] - y[np.newaxis, :]
            cov_matrix = np.exp(-0.5 * (dx ** 2 + dy ** 2) / l ** 2)
            return cov_matrix

        kernel_matrix = gaussian_kernel(node[:, 0], node[:, 1], PhyLoss.proj_l)
        kernel_matrix = torch.tensor(kernel_matrix).to(device).to(self.dtype)
        x = torch.tensor(node[:, 0]).to(device).to(self.dtype)
        y = torch.tensor(node[:, 1]).to(device).to(self.dtype)

        branch_input = torch.tensor(branch_input, dtype=torch.float64).to(device)
        branch_input = branch_input.detach()
        if self.max_iter == 0:
            branch_input = branch_input * (1 - x ** 2) * (1 - y ** 2) / 2
            net.train()
            return branch_input
        branch_input.requires_grad = True
        opt = torch.optim.Adam(params=[branch_input], lr=self.lr)
        net.eval()
        net.to(self.dtype)
        for _ in range(self.max_iter):
            tsam = (kernel_matrix @ branch_input.T / kernel_matrix.sum(dim=1, keepdim=True)).T
            tsam = tsam * (1 - x ** 2) * (1 - y ** 2) / 2
            opt.zero_grad()
            loss_phy = torch.mean(self.get_phy_loss(net, tsam, self.trunk_input, device) ** 2)
            loss_bc = torch.mean(self.get_bc_loss(net, tsam, self.trunk_input_bc, device) ** 2)
            loss_ic = torch.mean(self.get_ic_loss(net, tsam, self.trunk_input_ic, device) ** 2)
            loss = -1 * (loss_phy + loss_bc + loss_ic)
            loss.backward()
            opt.step()
        branch_input = (kernel_matrix @ branch_input.T / kernel_matrix.sum(dim=1, keepdim=True)).T
        branch_input = branch_input * (1 - x ** 2) * (1 - y ** 2) / 2
        net.train()
        return branch_input

    def update_training_samples(self, net, kept_num, train_dataset, batch_size=32):
        train_dataset.move_to("cpu")
        net.to(self.dtype)
        net.eval()
        branch_input = self.get_branch_input(net, kept_num=kept_num, device=self.device)
        u0_list = branch_input.detach().cpu().numpy()
        results_U = []
        del_idx = []

        nx = 64
        ny = 64
        nt = 10000
        recorded_nt = 100
        tmax = 1
        dt = tmax / nt
        mesh_scaler = 2
        x = np.linspace(-1, 1, nx)  # Coordinate Along X direction
        y = np.linspace(-1, 1, ny)  # Coordinate Along Y direction
        X, Y = np.meshgrid(x, y)
        orignal_node = np.stack((X.flatten(), Y.flatten()), axis=1)
        from create_dataset import solver_maccormack as solver
        from create_dataset import interpolation
        print('Start generating the solutions...')
        for i in range(len(u0_list)):
            tnx2 = nx*mesh_scaler
            tny2 = ny*mesh_scaler
            grid_X2 = np.linspace(0, 2, tnx2) - 1
            grid_Y2 = np.linspace(0, 2, tny2) - 1
            X2, Y2 = np.meshgrid(grid_X2, grid_Y2)
            updated_nodes = np.stack((X2.flatten(), Y2.flatten()), axis=1)
            u0_2 = interpolation(u0_list[i].reshape(1, -1), orignal_node, updated_nodes)
            u0_2 = u0_2.reshape(tnx2, tny2)
            U = solver(u0_2.copy(), nx=tnx2, ny=tny2, nt=nt, dt=dt, nu=PhyLoss.mu, recorded_interval=int(nt / recorded_nt))
            U = np.array(U)
            U = U[:, ::mesh_scaler, ::mesh_scaler]
            results_U.append(U.reshape(-1))
            if np.isnan(U).any():
                del_idx.append(i)
                print("Deleted sample %d." % i)

        print('Generating the solutions finished.')
        if len(del_idx) > 0:
            u0_list = np.delete(u0_list, del_idx, axis=0)
            results_U = np.delete(results_U, del_idx, axis=0)
            print("Deleted %d samples." % len(del_idx))
        train_dataset.move_to(self.device)
        train_dataset.branch_input = torch.cat([train_dataset.branch_input,
                                                  torch.tensor(u0_list, dtype=torch.float32).to(self.device)], dim=0)
        added_label_u = torch.tensor(results_U, dtype=torch.float32).to(self.device)
        train_dataset.label = torch.cat([train_dataset.label, added_label_u], dim=0)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        net.to(torch.float32)
        net.train()
        return train_dataset, train_loader

    @staticmethod
    def get_trunk_input(num=10000, device='cpu'):
        x = torch.rand(num, 1).to(device) * 2 - 1
        y = torch.rand(num, 1).to(device) * 2 - 1
        t = torch.rand(num, 1).to(device)
        return torch.cat([x, y, t], dim=1).to(device)

    @staticmethod
    def get_trunk_input_ic(device='cpu'):
        nx = ny = 64
        x = np.linspace(-1, 1, nx)  # Coordinate Along X direction
        y = np.linspace(-1, 1, ny)  # Coordinate Along Y direction
        X, Y = np.meshgrid(x, y)
        node = np.stack((X.flatten(), Y.flatten()), axis=1)
        t = torch.zeros(len(node), 1).to(device)
        node = torch.tensor(node).to(device)
        trunk_input_ic = torch.cat([node, t], dim=1)
        return trunk_input_ic

    @staticmethod
    def get_trunk_input_bc(num=200, device='cpu'):
        # At x= -1
        x = torch.ones(num, 1) * -1
        y = torch.linspace(-1, 1, num).view(-1, 1)
        t = torch.rand(num, 1)
        trunk_input_bc1 = torch.cat([x, y, t], dim=1)
        # At x = 1
        x = torch.ones(num, 1)
        y = torch.linspace(-1, 1, num).view(-1, 1)
        t = torch.rand(num, 1)
        trunk_input_bc2 = torch.cat([x, y, t], dim=1)
        # At y = -1
        x = torch.linspace(-1, 1, num).view(-1, 1)
        y = torch.ones(num, 1) * -1
        t = torch.rand(num, 1)
        trunk_input_bc3 = torch.cat([x, y, t], dim=1)
        # At y = 1
        x = torch.linspace(-1, 1, num).view(-1, 1)
        y = torch.ones(num, 1)
        t = torch.rand(num, 1)
        trunk_input_bc4 = torch.cat([x, y, t], dim=1)
        trunk_input_bc = torch.cat([trunk_input_bc1, trunk_input_bc2, trunk_input_bc3, trunk_input_bc4], dim=0)
        return trunk_input_bc.to(device)

    @staticmethod
    def get_phy_loss(net, branch_input, trunk_input, device='cpu'):
        dx = dy = 5e-3
        dt = 1e-3
        trunk_input_px1 = trunk_input + torch.tensor([dx, 0, 0], dtype=torch.float64).to(device)
        trunk_input_px2 = trunk_input + torch.tensor([dx * 2, 0, 0], dtype=torch.float64).to(device)
        trunk_input_mx1 = trunk_input - torch.tensor([dx, 0, 0], dtype=torch.float64).to(device)
        trunk_input_mx2 = trunk_input - torch.tensor([dx * 2, 0, 0], dtype=torch.float64).to(device)
        trunk_input_py1 = trunk_input + torch.tensor([0, dy, 0], dtype=torch.float64).to(device)
        trunk_input_py2 = trunk_input + torch.tensor([0, dy * 2, 0], dtype=torch.float64).to(device)
        trunk_input_my1 = trunk_input - torch.tensor([0, dy, 0], dtype=torch.float64).to(device)
        trunk_input_my2 = trunk_input - torch.tensor([0, dy * 2, 0], dtype=torch.float64).to(device)
        trunk_input_pt1 = trunk_input + torch.tensor([0, 0, dt], dtype=torch.float64).to(device)
        trunk_input_pt2 = trunk_input + torch.tensor([0, 0, dt * 2], dtype=torch.float64).to(device)
        trunk_input_mt1 = trunk_input - torch.tensor([0, 0, dt], dtype=torch.float64).to(device)
        trunk_input_mt2 = trunk_input - torch.tensor([0, 0, dt * 2], dtype=torch.float64).to(device)
        tpx1 = net.trunk_net(trunk_input_px1)
        tpx2 = net.trunk_net(trunk_input_px2)
        tmx1 = net.trunk_net(trunk_input_mx1)
        tmx2 = net.trunk_net(trunk_input_mx2)
        tpy1 = net.trunk_net(trunk_input_py1)
        tpy2 = net.trunk_net(trunk_input_py2)
        tmy1 = net.trunk_net(trunk_input_my1)
        tmy2 = net.trunk_net(trunk_input_my2)
        tpt1 = net.trunk_net(trunk_input_pt1)
        tpt2 = net.trunk_net(trunk_input_pt2)
        tmt1 = net.trunk_net(trunk_input_mt1)
        tmt2 = net.trunk_net(trunk_input_mt2)

        tu0 = net.trunk_net(trunk_input)

        u_x = (-tpx2 + 8 * tpx1 - 8 * tmx1 + tmx2) / (12 * dx)
        u_xx = (-tpx2 + 16 * tpx1 - 30 * tu0 + 16 * tmx1 - tmx2) / (12 * dx ** 2)
        u_y = (-tpy2 + 8 * tpy1 - 8 * tmy1 + tmy2) / (12 * dy)
        u_yy = (-tpy2 + 16 * tpy1 - 30 * tu0 + 16 * tmy1 - tmy2) / (12 * dy ** 2)

        u_t = (-tpt2 + 8 * tpt1 - 8 * tmt1 + tmt2) / (12 * dt)

        # Burgers' equation: u_t + u * (u_x + u_y) = mu (u_xx + u_yy)
        trunk_output_res = u_t - PhyLoss.mu * (u_yy + u_xx)
        branch_output = net.branch_net(branch_input)
        res1 = torch.matmul(branch_output, trunk_output_res.T)
        res2 = torch.matmul(branch_output, (u_x + u_y).T) * torch.matmul(branch_output, tu0.T)
        res = res1 + res2
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
