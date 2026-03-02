import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def add_results_to_file(file_path, content):
    with open(file_path, "a") as file:
        file.write("\n" + content)

class Evaluation():
    def __init__(self, device=None, freq=2, filename="results", ls_list=[0.4], is_transformed=True):
        if device is None:
            # Determine the device to be used for the training
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.frequency = freq
        self.filename = filename
        self.ls_list = ls_list
        self.is_transformed = is_transformed

    def get_u(self, pred_results, trunk_input):
        x = trunk_input[:, 0].view(-1, 1)
        t = trunk_input[:, 1].view(-1, 1)
        y = pred_results * x * (1 - x) * t
        return y

    def evaluate(self, net, epoch):
        return

    def get_results(self, net):
        results = []
        for ls in self.ls_list:
            results.append(self.get_results_single(net, ls))
        # Generate the formatted headline and error lines
        headline = "ls = " + " ".join(str(ls) for ls in self.ls_list)
        error_line = "Error: " + " ".join(str(result.item()) for result in results)

        # Write the lines to the file
        add_results_to_file(self.filename, headline)
        add_results_to_file(self.filename, error_line)
        return results

    def get_results_single(self, net, ls):
        filepath = "dataset"
        filename = filepath + r"\l=" + str(ls) + ".npz"
        datset = np.load(filename)
        branch_input = torch.tensor(datset["branch_input"], dtype=torch.float32).to(self.device)
        trunk_input = torch.tensor(datset["trunk_input"], dtype=torch.float32).to(self.device)
        y = datset["y"].reshape(-1, 1)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        pred_results = net.forward_branch_trunk_fixed(branch_input, trunk_input)
        if self.is_transformed:
            pred_results = self.get_u(pred_results, trunk_input.repeat(len(branch_input), 1))
        relative_L2_error = torch.nn.MSELoss()(pred_results, y) / torch.nn.MSELoss()(y, torch.zeros_like(y))
        return relative_L2_error

    def visual_results(self, net, ls):
        filepath = "dataset"
        filename = filepath + r"\l=" + str(ls) + ".npz"
        dataset = np.load("kmin0.1b0.2dataset2.npz")
        branch_input = torch.tensor(dataset["branch_input"][0], dtype=torch.float32).to(self.device).view(1, -1)
        trunk_input = torch.tensor(dataset["trunk_input"], dtype=torch.float32).to(self.device)
        y = dataset["y"].reshape(-1, 1)
        y = y[: len(trunk_input)]
        pred_results = net.forward_branch_trunk_fixed(branch_input, trunk_input)
        if self.is_transformed:
            pred_results = self.get_u(pred_results, trunk_input.repeat(len(branch_input), 1))
        # Visualize via the matplotlib contour plot using three 3 subplots
        plt.figure(figsize=(15, 4.5))
        plt.subplot(1, 3, 1)
        # Plot the ground truth using  trunk_input and y
        branch_input = branch_input.cpu().detach().numpy()
        trunk_input = trunk_input.cpu().detach().numpy()
        # y = y.cpu().detach().numpy()
        pred_results = pred_results.cpu().detach().numpy()
        plt.contourf(trunk_input[:, 0].reshape(-1, 100), trunk_input[:, 1].reshape(-1, 100), y.reshape(-1, 100), levels=100, cmap='coolwarm')
        plt.title("Ground Truth")
        plt.colorbar()
        plt.subplot(1, 3, 2)
        # Plot the prediction using trunk_input and pred_results
        plt.contourf(trunk_input[:, 0].reshape(-1, 100), trunk_input[:, 1].reshape(-1, 100), pred_results.reshape(-1, 100), levels=100, cmap='coolwarm')
        plt.title("Prediction")
        plt.colorbar()
        plt.subplot(1, 3, 3)
        # Plot the error using trunk_input and y - pred_results
        plt.contourf(trunk_input[:, 0].reshape(-1, 100), trunk_input[:, 1].reshape(-1, 100), (y - pred_results).reshape(-1, 100), levels=100, cmap='coolwarm')
        rel_error = np.linalg.norm(y - pred_results) / np.linalg.norm(y)
        plt.text(0.6, -0.95, f"Rel. $L_2$ Error: {rel_error:.2e}", horizontalalignment='center',
                 verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))
        plt.colorbar()
        plt.title("Error")
        plt.show()

if __name__ == '__main__':
    # Evaluation()
    # Evaluation().plot_results(None)
    # acc_results = pd.read_csv('dataset2_output.csv')
    # acc_results = torch.tensor(acc_results.values, dtype=torch.float32)
    # acc_results = acc_results.reshape(-1, 1)
    # # Calculate the mean square value
    # print(torch.nn.MSELoss()(acc_results, torch.zeros_like(acc_results)))
    net_name = r"150width_results_rar_trainable.pth"
    net = torch.load(net_name)
    evaluation = Evaluation()
    evaluation.visual_results(net, 3)