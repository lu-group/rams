#############################################################################
# Standard Physics Informed Neural Network Training process
# The Adam optimizer is used to optimize the loss function
#############################################################################
import torch
from tqdm import *
# from src.IO.writer.writer_results import save_net as save_net
import torch.optim as optim

def train(training, loss, net, device=None, evaluation=None):
    # param net: the neural networks model needed to be trained
    # param training_info: the training information
    # return: a trained net

    # Turn on the training mode
    net.train()
    if device is None:
        # Determine the device to be used for the training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    # Decode the training information from the model.training dictionary
    try:
        max_epochs = training["MaxEpochs"]
        learning_rate = training["LearningRate"]
    except KeyError:
        raise Exception("Training information is not defined in the input file")
    try:
        opt_reset_interval = training["OptResetInterval"]
        max_reset_epoch = training["MaxResetEpoch"]
        is_opt_reset = True
    except KeyError:
        is_opt_reset = False
    # Define the optimizer
    opt = torch.optim.Adam(params=net.parameters(), lr=learning_rate)

    # Training loop
    desc = "start training..."
    pbar = tqdm(range(max_epochs), desc=desc)
    optnet = net
    minloss = 100000
    last_save_epoch = 100

    if evaluation is not None:
        eva_freq = evaluation.frequency
    else:
        eva_freq = None
    for epoch in pbar:
        if is_opt_reset and epoch % opt_reset_interval == 0 and epoch < max_reset_epoch:
            opt = torch.optim.Adam(params=net.parameters(), lr=learning_rate)
        # Update the loss functoin
        loss.update_losses(net, epoch)
        loss_value = 0
        # In the standard PI training method, the loss terms are simply added together.
        for lossitem in loss.losses:
            loss_value += lossitem
        # Zero the gradients
        opt.zero_grad()
        # Backpropagation
        loss_value.backward()
        # Update the model parameters
        opt.step()
        if eva_freq is not None and epoch % eva_freq == 0:
            evaluation.evaluate(net, epoch)

        # Update the progress bar
        qbar_description = f"Epoch {epoch + 1}/{max_epochs}, Loss: {loss_value:.3e}, MinLoss: {minloss:.3e}, Loss Terms:"
        # Add the loss item value to qbar
        for lossitem in loss.losses:
            qbar_description += f", {lossitem.item():.3e}"
        pbar.set_description(qbar_description)

        if loss_value.item() < minloss and epoch > max_epochs - 10000:
            minloss = loss_value.item()
            optnet = net

    if not training["isLBFGS"]:
        return optnet
    desc = "L-BFGS fine-tuning..."
    lbfgs_max_epochs = training["LBFGSMaxEpochs"]
    pbar = tqdm(range(lbfgs_max_epochs), desc=desc)
    opt = optim.LBFGS(net.parameters(), lr=1.0, max_iter=20, history_size=10)
    for epoch in pbar:
        # if torch.isnan(loss.losses[0]):
        #     return last_net
        def closure():
            opt.zero_grad()
            # Update the loss functoin
            loss.update_losses(net, -1)
            loss_value = 0
            # In the standard PI training method, the loss terms are simply added together.
            for lossitem in loss.losses:
                loss_value += lossitem
            loss_value.backward()
            return loss_value
        # Update the model parameters
        opt.step(closure)
        qbar_description = f"Epoch {epoch + 1}/{lbfgs_max_epochs}, Loss: {loss_value:.3e}, MinLoss: {minloss:.3e}, Loss Terms:"
        # Add the loss item value to qbar
        for lossitem in loss.losses:
            qbar_description += f", {lossitem.item():.3e}"
        pbar.set_description(qbar_description)

        if loss_value.item() < minloss:
            minloss = loss_value.item()
            optnet = net
    evaluation.evaluate(net, epoch)
    return net
