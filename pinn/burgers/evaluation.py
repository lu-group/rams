import torch
import numpy as np
import matplotlib.pyplot as plt
from fdm_solver import fdm_burgers
class Evaluation_Burgers():
    def __init__(self, mu, freq, device=None, is_net_transformed=True):
        if device is None:
            # Determine the device to be used for the training
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.mu = mu
        self.frequency = freq
        self.t, self.x, self.u = fdm_burgers(mu)
        # benchmark_solution = get_testdata()
        # self.x = np.array([r[0] for r in benchmark_solution])
        # self.t = np.array([r[1] for r in benchmark_solution])
        # self.u = np.array([r[2] for r in benchmark_solution])
        self.x = self.x.flatten()
        self.t = self.t.flatten()
        self.u = self.u.flatten()
        self.x = torch.tensor(self.x, dtype=torch.float32).view(-1, 1)
        self.t = torch.tensor(self.t, dtype=torch.float32).view(-1, 1)
        self.u = torch.tensor(self.u, dtype=torch.float32).view(-1, 1)
        self.x, self.t, self.u = self.x.to(self.device), self.t.to(self.device), self.u.to(self.device)
        self.results = []
        self.is_net_transformed = is_net_transformed

    def get_u(self, net, sam):
        is_net_transformed = self.is_net_transformed
        if is_net_transformed:
            output = net(sam)
            u = (-torch.sin(torch.pi * sam[:, 0].view(-1, 1))
                 + (1 - (sam[:, 0] ** 2).view(-1, 1)) * sam[:, 1].view(-1, 1) * output)
        else:
            u = net(sam)
        return u

    def evaluate(self, net, epoch):
        net.eval()
        u_pre = self.get_u(net, torch.cat((self.x, self.t), 1))
        # u_pre = net(torch.cat((self.x, self.t), 1))
        delta_u = self.u - u_pre
        # Relative L2 error
        error = torch.norm(delta_u) / torch.norm(self.u)
        self.results.append([epoch, error])

    def plot_results(self, net, is_show=True):
        data = np.load("Burgers.npz")
        t, x, exact = data["t"], data["x"], data["usol"].T
        xx, tt = np.meshgrid(x, t)
        y_exact = exact.flatten()[:, None]
        y_exact = y_exact.reshape(tt.shape)
        # Transform xx and tt to net input tensor
        net_input = torch.tensor(np.vstack((np.ravel(xx), np.ravel(tt))).T, dtype=torch.float32).to(self.device)
        y_pred = self.get_u(net, net_input).cpu().detach().numpy().reshape(y_exact.shape)

        base_fontsize = 16

        fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
        import matplotlib.ticker as ticker
        # 子图0：真实解
        c0 = axs[0].contourf(xx, tt, y_exact, levels=50, cmap='coolwarm')
        cb0 = fig.colorbar(c0, ax=axs[0])
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb0.locator = tick_locator
        cb0.update_ticks()
        # axs[0].set_title('Ground Truth', fontsize=base_fontsize)  # 设置标题字体大小
        # 设置刻度标签字体大小
        axs[0].tick_params(axis='both', which='major', labelsize=base_fontsize)

        # 子图1：预测解
        c1 = axs[1].contourf(xx, tt, y_pred, levels=50, cmap='coolwarm')
        cb1 = fig.colorbar(c1, ax=axs[1])
        cb1.locator = ticker.MaxNLocator(nbins=5)
        cb1.update_ticks()
        # axs[1].set_title('Trainable RAR', fontsize=base_fontsize)
        axs[1].tick_params(axis='both', which='major', labelsize=base_fontsize)

        # 子图2：误差图
        c2 = axs[2].contourf(xx, tt, np.abs(y_exact - y_pred), levels=50, cmap='coolwarm')
        cb2 = fig.colorbar(c2, ax=axs[2])
        cb2.ax.locator_params(nbins=5)
        # axs[2].set_title('Error', fontsize=base_fontsize)
        axs[2].tick_params(axis='both', which='major', labelsize=base_fontsize)

        # 设置colorbar的刻度标签字体大小
        for cb in [cb0, cb1, cb2]:
            cb.ax.tick_params(labelsize=base_fontsize)  # 也可以单独设置每个colorbar

        # 计算误差并添加文本
        delta_u = np.abs(y_exact - y_pred)
        error = np.linalg.norm(delta_u) / np.linalg.norm(y_exact)

        for ax in axs:
            ax.set_xticks(np.linspace(ax.get_xlim()[0], 1, 3))
            ax.set_yticks(np.linspace(ax.get_ylim()[0], 1, 3))

        if is_show:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()
        return error

    def output(self):
        print("Epoch, Relative L2 Error")
        # Print the results in two effective digits
        for result in self.results:
            print(f"{result[0]}, {result[1]:.2e}")
        return self.results

# def get_testdata():
#     data = np.load("Burgers.npz")
#     t, x, exact = data["t"], data["x"], data["usol"].T
#     xx, tt = np.meshgrid(x, t)
#     X = np.vstack((np.ravel(xx), np.ravel(tt))).T
#     y = exact.flatten()[:, None]
#     y_reshaped = y.reshape(tt.shape)  # Reshape y to 2D if it's flattened
#     # Plot the contour
#     # fig, ax = plt.subplots()
#     # c = ax.contourf(xx, tt, y_reshaped, levels=50, cmap='viridis')
#     # fig.colorbar(c, ax=ax)
#     # plt.show()
#     # Prepare results list
#     results = []
#     for i in range(len(xx)):
#         for j in range(len(xx[i])):
#             results.append([xx[i][j], tt[i][j], y_reshaped[i][j]])
#     return results
# results = get_testdata()
# x = np.array([r[0] for r in results])
# t = np.array([r[1] for r in results])
# y = np.array([r[2] for r in results])
#
# plt.figure()
# scatter = plt.scatter(x, t, c=y, cmap='viridis')
# plt.colorbar(scatter)
# plt.title('2D Distribution of u(x,y)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

# mu = 0.01 / np.pi
# t, x, u = fdm_burgers(mu)
# fig, ax = plt.subplots()
# c = ax.contourf(x, t, u, levels=50, cmap='viridis')
# fig.colorbar(c, ax=ax)
# plt.show()

# Plot the exact solution; X is a n*2 array; y is the exact solution

