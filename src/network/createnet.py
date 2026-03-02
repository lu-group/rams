from src.network.fcnn import FCNN
from src.network.resfcnn import ResidualFCNN
from src.network.activationfun.getactfunc import getactfn

# Create networks from the given information
def createnet(networkinfo):
    try:
        net_name = networkinfo["Name"]
        net_type = networkinfo["Type"]
    except:
        print("Error: The network information is not complete.")
        return None
    if net_type == "FCNN":
        try:
            inputsize = networkinfo["InputSize"]
            outputsize = networkinfo["OutputSize"]
            hiddensizes = networkinfo["HiddenSizes"]
            act_fn = networkinfo["ActivationFunc"]
            name = networkinfo["Name"]

            if type(act_fn) != type([]):
                tact_fn = act_fn
                act_fn = []
                for ii in range(len(hiddensizes)):
                    act_fn.append(tact_fn)
            # Transfer the string to the torch function
            act_fn = getactfn(act_fn)
            net = FCNN(act_fn=act_fn, input_size=inputsize, output_size=outputsize, hidden_sizes=hiddensizes, name=name)
        except:
            print("Error: The network information is not complete.")
            return None
    elif net_type == "ResFCNN":
        try:
            inputsize = networkinfo["InputSize"]
            outputsize = networkinfo["OutputSize"]
            hiddensizes = networkinfo["HiddenSizes"]
            act_fn = networkinfo["ActivationFunc"]
            name = networkinfo["Name"]
            if type(act_fn) != type([]):
                tact_fn = act_fn
                act_fn = []
                for ii in range(len(hiddensizes)):
                    act_fn.append(tact_fn)
            # Transfer the string to the torch function
            act_fn = getactfn(act_fn)
            net = ResidualFCNN(act_fn=act_fn, input_size=inputsize, output_size=outputsize, hidden_sizes=hiddensizes,
                               name=name)
        except:
            print("Error: The network information is not complete.")
            return None
    net.name = net_name
    print("Network created!")
    return net

if __name__ == '__main__':
    networkinfo = {"Name": "test", "Type": "FCNN", "ActivationFunc": "Tanh", "InputSize": 24, "OutputSize": 24, "HiddenSizes": [128, 128, 128]}
    net = createnet(networkinfo)
    print(net)
    print(net.name)