import numpy as np
import matplotlib.pyplot as plt
import torch

def test_results(net_name, dataset_name, device=None, is_transformed=True, num_x=50, num_y=50):
    x_len = 2.0
    y_len = 2.0
    num_x = num_x - 1
    num_y = num_y - 1
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = torch.load(net_name, map_location=device)
    dataset = np.load(dataset_name)
    branch_input = dataset['branch_input']
    trunk_input = dataset['trunk_input']
    u_solution = dataset['u_solution'] #/ ele_area
    branch_input = torch.tensor(branch_input, dtype=torch.float32).to(device)
    trunk_input = torch.tensor(trunk_input, dtype=torch.float32).to(device)
    u_solution = torch.tensor(u_solution, dtype=torch.float32).to(device)
    branch_output = net.branch_net(branch_input)
    trunk_output = net.trunk_net(trunk_input)
    if is_transformed:
        x = trunk_input[:, 0].view(-1, 1)
        y = trunk_input[:, 1].view(-1, 1)
        x = x.repeat(1, trunk_output.shape[1])
        y = y.repeat(1, trunk_output.shape[1])
        trunk_output = trunk_output * (1 - x ** 2) * (1 - y ** 2)
    u_pred = torch.matmul(branch_output, trunk_output.transpose(0,1))
    u_pred = u_pred.cpu().detach().numpy()
    u_solution = u_solution.cpu().detach().numpy()
    relative_l2_error = np.linalg.norm(u_pred - u_solution) / np.linalg.norm(u_solution)
    print("Relative L2 error: ", relative_l2_error)
    return relative_l2_error

def visual_results(net_name, dataset_name, device=None, mesh_num=50, is_transformed=True, num_x=50, num_y=50):
    x_len = 2.0
    y_len = 2.0
    num_x = num_x - 1
    num_y = num_y - 1
    ele_area = 1 / num_x / num_y
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_idx = 0
    net = torch.load(net_name, map_location=device)
    dataset = np.load(dataset_name)
    branch_input = dataset['branch_input'] #/ ele_area
    trunk_input = dataset['trunk_input']
    u_solution = dataset['u_solution']
    branch_input = torch.tensor(branch_input, dtype=torch.float32).to(device)
    trunk_input = torch.tensor(trunk_input, dtype=torch.float32).to(device)
    u_solution = torch.tensor(u_solution, dtype=torch.float32).to(device)[text_idx]
    branch_output = net.branch_net(branch_input)
    trunk_output = net.trunk_net(trunk_input)
    if is_transformed:
        x = trunk_input[:, 0].view(-1, 1)
        y = trunk_input[:, 1].view(-1, 1)
        x = x.repeat(1, trunk_output.shape[1])
        y = y.repeat(1, trunk_output.shape[1])
        trunk_output = trunk_output * (1 - x ** 2) * (1 - y ** 2)
    u_pred = torch.matmul(branch_output, trunk_output.transpose(0, 1))[text_idx]
    u_pred = u_pred.cpu().detach().numpy()
    u_solution = u_solution.cpu().detach().numpy()
    relative_l2_error = np.linalg.norm(u_pred - u_solution) / np.linalg.norm(u_solution)
    print("Relative L2 error: ", relative_l2_error)
    grid_x, grid_y = trunk_input[:, 1].cpu().detach().numpy().reshape(mesh_num, mesh_num), trunk_input[:, 0].cpu().detach().numpy().reshape(mesh_num, mesh_num)
    # Plot using three contours
    fig, ax = plt.subplots(1, 4, figsize=(24, 5))
    c1 = ax[0].contourf(grid_x, grid_y, u_solution.reshape(mesh_num, mesh_num), 100, cmap='jet')
    plt.colorbar(c1, ax=ax[0])  # Add colorbar for the exact solution
    ax[0].set_title("Exact solution")

    # Plot predicted solution
    c2 = ax[1].contourf(grid_x, grid_y, u_pred.reshape(mesh_num, mesh_num), 100, cmap='jet')
    plt.colorbar(c2, ax=ax[1])  # Add colorbar for the predicted solution
    ax[1].set_title("Predicted solution")

    # Plot error (difference between predicted and exact solution)
    error = (u_pred - u_solution).reshape(mesh_num, mesh_num)
    c3 = ax[2].contourf(grid_x, grid_y, error, 100, cmap='jet')
    plt.colorbar(c3, ax=ax[2])  # Add colorbar for the error
    ax[2].set_title("Error")

    # Plot the source term (branch input)
    c4 = ax[3].contourf(grid_x, grid_y, branch_input[text_idx].cpu().detach().numpy().reshape(mesh_num, mesh_num), 100, cmap='jet')
    plt.colorbar(c4, ax=ax[3])  # Add colorbar for the source term
    ax[3].set_title("Source term")

    plt.tight_layout()
    plt.show()
    return relative_l2_error

def run_eval():
    dataset_name = "dataset2.npz"
    results = []
    print("=" * 20 + "RAR (without RAMS)" + "=" * 20)
    for i in [100, 200, 300, 400, 600, 800]:
        print("Sample number: ", i, end="; ")
        net_name = r"models/" + str(i) + "sam_random.pth"
        tresults = test_results(net_name, dataset_name)
        results.append(tresults)

    print("=" * 20 + "RAR (with RAMS)" + "=" * 20)
    for i in [100, 200, 300, 400, 600, 800]:
        print("Sample number: ", i, end="; ")
        net_name = r"models/" + str(i) + "sam_random_trainable_300iter.pth"
        # net_name = r"models/" + str(i) + "sam_rar.pth"
        # net_name = r"models/" + str(i) + "sam_rar_trainable_300iter.pth"
        tresults = test_results(net_name, dataset_name)
        results.append(tresults)

    print("=" * 20 + "Random sampling (without RAMS)" + "=" * 20)
    for i in [100, 200, 300, 400, 600, 800]:
        print("Sample number: ", i, end="; ")
        net_name = r"models/" + str(i) + "sam_rar.pth"
        tresults = test_results(net_name, dataset_name)
        results.append(tresults)

    print("=" * 20 + "Random sampling (with RAMS)" + "=" * 20)
    for i in [100, 200, 300, 400, 600, 800]:
        print("Sample number: ", i, end="; ")
        net_name = r"models/" + str(i) + "sam_rar_trainable_300iter.pth"
        tresults = test_results(net_name, dataset_name)
        results.append(tresults)

if __name__ == "__main__":
    dataset_name = "dataset2.npz"
    results = []
    for i in [100,200,300,400,600,800]:
        # net_name = r"models/" + str(i) + "sam_random.pth"
        # net_name = r"models/" + str(i) + "sam_random_trainable_300iter.pth"
        # net_name = r"models/" + str(i) + "sam_rar.pth"
        net_name = r"models/" + str(i) + "sam_rar_trainable_300iter.pth"
        tresults = test_results(net_name, dataset_name)
        results.append(tresults)
    for result in results:
        print(result)
    # visual_results(net_name, dataset_name)
