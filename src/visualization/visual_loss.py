"""
For the visualization of the loss during the training process.
An example about how to use this class is provided in beam2D/beam_elastic_linear01/loss_random.py.
"""

import matplotlib.pyplot as plt
import numpy as np

class Visual_loss():
    def __init__(self, loss_term_num, max_epoch, update_freq=10, pause_time=0.01):
        self.loss_term_num = loss_term_num
        self.max_epoch = max_epoch
        self.update_freq = update_freq
        self.losses_list = []
        self.epoch_list = []
        self.pause_time = pause_time

    def plot_losses(self, current_epoch, losses, filename='iterations-loss.png'):
        """
        Update the loss curve during the training process.
        """
        if len(losses) != self.loss_term_num:
            raise ValueError("The length of the loss list is not equal to the number of loss terms.")

        # Convert losses to numpy and update the lists
        losses_value = np.array([loss.detach().numpy() for loss in losses]).reshape(1, -1)
        if len(self.losses_list) == 0:
            self.losses_list = losses_value
        else:
            self.losses_list = np.vstack((self.losses_list, losses_value))
        self.epoch_list.append(current_epoch)

        # Plot update condition
        if current_epoch % self.update_freq == 0 or current_epoch == self.max_epoch:
            plt.figure(0)
            plt.clf()  # Clear the current figure to update the plot

            for i in range(self.loss_term_num):
                tloss = self.losses_list[:, i]
                plt.plot(self.epoch_list, tloss, label=f'Loss_{i + 1}')

            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.yscale('log')
            plt.xlim(0, self.max_epoch)
            plt.legend(loc='upper right', fontsize=12)
            plt.grid(True)
            plt.tight_layout()

            # Show plot in a non-blocking way and save on the last epoch
            plt.draw()
            plt.pause(self.pause_time)  # Pause to ensure the plot updates
            if current_epoch == self.max_epoch:
                plt.savefig(filename)
                plt.ioff()  # Turn off the interactive mode
                plt.show()  # Show the final plot in blocking mode