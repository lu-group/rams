# Visualization of the residuals
import matplotlib.pyplot as plt
import imageio
import os

class Visual_residual():
    def __init__(self, row_num, column_num):
        """
        Inputs:
            row_num: int, the number of rows of the subfigures
            column_num: int, the number of columns of the subfigures
        """
        self.row_num = row_num
        self.column_num = column_num
        self.filename_list = []
        self.fig = self.initial_residual_fig()

    def initial_residual_fig(self):
        plt.ion()
        plt.rc('font', size=8)  # controls default text sizes
        plt.rc('axes', titlesize=8)  # fontsize of the axes title
        plt.rc('axes', labelsize=8)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=8)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=8)  # fontsize of the tick labels
        plt.rc('legend', fontsize=8)  # legend fontsize
        plt.rc('figure', titlesize=8)  # fontsize of the figure title
        plt.rcParams["font.family"] = "Times New Roman"
        fig = plt.figure(figsize=(8, 10))
        return fig

    def plot_resdiual(self, residual_list, current_epoch, max_epoch, gif_name='residual.gif'):
        """
        Inputs:
            residual_list: list, [x, y, the residual data]
            current_epoch: int, the current epoch
            max_epoch: int, the maximum epoch
        """
        plt.figure(1)
        plt.clf()
        for i in range(len(residual_list)):
            # update the data
            x = residual_list[i][0]
            y = residual_list[i][1]
            residual = residual_list[i][2]
            x = x.detach().numpy().reshape(-1, 1)
            y = y.detach().numpy().reshape(-1, 1)
            residual = residual.detach().numpy().flatten()

            # update the plot
            ax = self.fig.add_subplot(self.row_num, self.column_num, i+1)
            scatter = ax.scatter(x, y, c=residual, cmap='jet', s=2)
            plt.colorbar(scatter)

        plt.suptitle(f'Visualization of residuals at sampling points at epoch={current_epoch}', fontsize=12)
        plt.tight_layout()
        self.fig.canvas.draw()
        plt.pause(1e-5)

        filename = f"epoch_{current_epoch}.png"
        plt.savefig(filename)
        self.filename_list.append(filename)

        if current_epoch == max_epoch:
            images = []
            for file in self.filename_list:
                images.append(imageio.imread(file))
                os.remove(file)
            imageio.mimsave(gif_name, images, fps=10)