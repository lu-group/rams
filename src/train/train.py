import src.train.standard_physics_informed as standard_physics_informed
# import src.train.adaptive_physics_informed as adaptive_physics_informed

def run_train(training, loss, net, device=None, evaluation=None):
    train_type = training["Type"]
    # print("Training type: %s" % train_type)
    if train_type == "StandardPI":
        # Standard physics-informed training
        net = standard_physics_informed.train(training, loss, net, device=device, evaluation=evaluation)
        # raise Exception("Standard physics-informed training is not implemented yet!")
    elif train_type == "AdaptivePI":
        # Adaptive physics-informed training
        # net = adaptive_physics_informed.train(training, loss, net, evaluation=evaluation)
        raise Exception("Adaptive physics-informed training is not implemented yet!")
    else:
        raise Exception("The training type is not supported!")
    print("Training finished!")
    return net