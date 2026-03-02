import torch, os
from src.util.logging import printlog
def save_net(net, is_log=True):
    filepath = "results" #training_info["ResultPath"]
    # Check if the path exists
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    net_name = net.config["name"]
    # Join the path and the name of the network
    filepath = os.path.join(filepath, net_name+".pth")
    torch.save(net, filepath)
    printlog("Network saved to: " + filepath, is_log=is_log)
