import torch

class LossBase():

    def __init__(self):
        pass

    def updated_losses(self):
        Exception("Function (updated_losses) is not defined in the loss class. Please built it with the output of a list"
                  " containing all the loss terms.")