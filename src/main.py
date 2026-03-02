from src.train.train import run_train as run_train
from src.IO.writer.writer_results import save_net as save_net

def run(training, loss, net, device=None, evaluation=None):

    # Train the network
    net = run_train(training, loss, net, device=device, evaluation=evaluation)
    # Set the network to evaluation mode
    net.eval()
    # Save the trained network
    # save_net(net)
    return net

