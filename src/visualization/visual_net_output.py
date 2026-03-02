# Visualization of the output of the network

import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize(net, model):
    visual_type = model.results["Visualization"]
    if visual_type == "1D":
        visualize_1D(net, model)
    elif visual_type == "2D":
        visualize_2D(net, model)
    else:
        raise ValueError("Visualization type not supported.")

def visualize_1D(net, model, samplenum=20):
    visualization_range = model.results["VisualizationRange"]

    # Uniformly sample points in the geometry
    x_values = np.linspace(visualization_range[0], visualization_range[1], samplenum)

    input_tensor = torch.tensor(x_values.reshape(-1, 1), dtype=torch.float32)
    output_tensor = net(input_tensor)
    output = output_tensor.detach().numpy()

    for i in range(output.shape[1]):  # Assuming output.shape[1] is the number of output variables
        fig_title = "1D Visualization of " + net.name + str(int(i + 1)) + "th output variable"
        visualize_1D_single_variable(x_values, output[:, i], title=fig_title)


def visualize_1D_single_variable(x_values, output, title=""):
    plt.figure(0)
    plt.plot(x_values, output, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('Output')
    plt.show()


def visualize_2D(net, model, samplenum=40):
    geometry = model.geometry  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], rectangle
    # Get the lower and upper bounds of the geometry
    low_bounds = np.min(geometry, axis=0)
    high_bounds = np.max(geometry, axis=0)

    # Uniformly sample points in the geometry
    x_values = np.linspace(low_bounds[0], high_bounds[0], samplenum)
    y_values = np.linspace(low_bounds[1], high_bounds[1], samplenum)
    X, Y = np.meshgrid(x_values, y_values)
    sample_points = np.vstack([X.ravel(), Y.ravel()]).T

    input_tensor = torch.tensor(sample_points, dtype=torch.float32)
    output_tensor = net(input_tensor)
    output = output_tensor.detach().numpy().T

    for i in range(len(output)):
        fig_title = "2D Visualization of " + net.name + str(int(i + 1)) + "th output variable"
        visualize_2D_single_variable(X, Y, output[i], title=fig_title, samplenum=samplenum)
        filename = f"output_{str(int(i + 1))}.png"
        plt.savefig(filename)

def visualize_2D_single_variable(X, Y, output, title="", samplenum=20):
    plt.figure(0)
    plt.clf()
    # Reshape the output to match the grid
    Z = output.reshape(X.shape)

    # Create the contour plot
    plt.contourf(X, Y, Z, samplenum, cmap='jet')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')


if __name__ == '__main__':
    netname = r"C:\Users\87618\PycharmProjects\pinnsa\examples\examples\shell_01_results\shell_testnet"
    net = torch.load(netname)
    class Model:
        pass
    model = Model()
    model.geometry = [[0, 0], [1, 0], [1, 1], [0, 1]]
    visualize_2D(net, model, samplenum=20)



