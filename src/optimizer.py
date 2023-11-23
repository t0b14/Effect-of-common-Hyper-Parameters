import torch.optim as optim

# currently only adam implemented
def optimizer_creator(weights, params):
    if params["optimizer_name"] == "adam":
        return optim.Adam(
            weights,
            lr=float(params["lr"]),
            betas=params.get("betas", (0.9, 0.999)),
            eps=params.get("eps", 1e-7),
            weight_decay=params.get("weight_decay", 0.0),
            amsgrad=params.get("amsgrad", False),
        )
    elif params["optimizer_name"] == "SGD":
        return optim.SGD(
            weights, 
            lr=float(params["lr"]), 
            momentum=params["momentum"]
            )
    elif params["optimizer_name"] == "adagrad":
        return optim.Adagrad(
            weights,
            lr=float(params["lr"]), 
            lr_decay = 1e-5,
            weight_decay = 1e-5,
            initial_accumulator_value = 0,
            eps = 1e-10
            )
    else:
        raise ValueError("Invalid optimizer name")