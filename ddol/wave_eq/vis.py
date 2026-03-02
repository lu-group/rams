import torch
import numpy as np

def evaluation_val(net, dataset):
    branch_input = dataset["branch_input"]
    trunk_input = dataset["trunk_input"]
    label = dataset["results"]
    branch_output = net.branch_net(torch.tensor(branch_input, dtype=torch.float32))
    trunk_output = net.trunk_net(torch.tensor(trunk_input, dtype=torch.float32))
    pred = torch.matmul(branch_output, trunk_output.T).detach().numpy()
    diff_norm = np.linalg.norm(pred - label, axis=1) / np.linalg.norm(label, axis=1)
    mean_error = np.mean(diff_norm)
    # loss = np.mean((pred - label) ** 2) / np.mean(label ** 2)
    # print("The loss is %.4e" % mean_error)
    return mean_error

if __name__ == '__main__':
    netname = "trainable_rar_sample_iter_200_repeat_1.pth"
    net = torch.load(netname, map_location=torch.device('cpu'))
    dataset = np.load(r"datasets\\testing_dataset2.npz")
    branch_input = dataset["branch_input"]
    trunk_input = dataset["trunk_input"]
    label = dataset["results"]
    branch_output = net.branch_net(torch.tensor(branch_input, dtype=torch.float32))
    trunk_output = net.trunk_net(torch.tensor(trunk_input, dtype=torch.float32))
    pred = torch.matmul(branch_output, trunk_output.T).detach().numpy()
    trunk_input = dataset["trunk_input"]
    grid_x = trunk_input[:, 0].reshape(1001, 201)
    grid_t = trunk_input[:, 1].reshape(1001, 201)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    # plt.figure(figsize=(15, 12))
    # plt.subplot(3, 1, 1)
    # # For the first subplot, we plot the ground truth
    # plt.contourf(grid_t, grid_x, label.reshape(1001, 201), 100, cmap='coolwarm')
    # cbar = plt.colorbar()
    # cbar.locator = MaxNLocator(7)
    # cbar.update_ticks()
    # plt.title("Ground truth")
    # plt.subplot(3, 1, 2)
    # # For the second subplot, we plot the prediction
    # plt.contourf(grid_t, grid_x, pred.reshape(1001, 201), 100, cmap='coolwarm')
    # cbar = plt.colorbar()
    # cbar.locator = MaxNLocator(7)
    # cbar.update_ticks()
    # plt.title("Prediction")
    # plt.subplot(3, 1, 3)
    # # For the third subplot, we plot the absolute error
    # plt.contourf(grid_t, grid_x, np.abs(pred - label).reshape(1001, 201), 100, cmap='coolwarm')
    # cbar = plt.colorbar()
    # rel_error = np.linalg.norm(pred - label) / np.linalg.norm(label)
    # plt.text(3, 0.05, f"Rel. $L_2$ Error: {rel_error:.2e}", horizontalalignment='center',
    #          verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))
    # cbar.locator = MaxNLocator(7)
    # cbar.update_ticks()
    # plt.title("Error")
    # plt.tight_layout()
    # plt.show()


    # Set the global font size
    plt.rcParams.update({'font.size': 16})

    plt.figure(figsize=(4,3))
    branch_input = branch_input.reshape(-1)
    plt.plot(np.linspace(0, 1, len(branch_input)), np.array(branch_input))
    # Set x_axis range(0,1)
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()
    # plt.close()

    plt.figure(figsize=(10, 8))


    # Function to set ticks for x and y axes and colorbar
    def set_ticks(ax, cbar):
        ax.xaxis.set_major_locator(MaxNLocator(8))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        cbar.locator = MaxNLocator(4)
        cbar.update_ticks()

        # For the first subplot, we plot the ground truth


    plt.subplot(3, 1, 1)
    contour = plt.contourf(grid_t, grid_x, label.reshape(1001, 201), 100, cmap='coolwarm')
    cbar = plt.colorbar(contour)
    ax = plt.gca()
    set_ticks(ax, cbar)
    # plt.title("Ground  truth")

    # For the second subplot, we plot the prediction
    plt.subplot(3, 1, 2)
    contour = plt.contourf(grid_t, grid_x, pred.reshape(1001, 201), 100, cmap='coolwarm')
    cbar = plt.colorbar(contour)
    ax = plt.gca()
    set_ticks(ax, cbar)
    # plt.title("Prediction")

    # For the third subplot, we plot the absolute error
    plt.subplot(3, 1, 3)
    contour = plt.contourf(grid_t, grid_x, np.abs(pred - label).reshape(1001, 201), 100, cmap='coolwarm')
    cbar = plt.colorbar(contour)
    rel_error = np.linalg.norm(pred - label) / np.linalg.norm(label)
    # plt.text(3, 0.05, f"Rel. $L_2$ Error: {rel_error:.2e}", horizontalalignment='center',
    #          verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))
    ax = plt.gca()
    set_ticks(ax, cbar)
    # plt.title("Error")

    # plt.tight_layout()
    plt.show()
