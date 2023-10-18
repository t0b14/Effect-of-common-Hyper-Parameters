import numpy as np
import torch.optim as optim


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
    # elif params["optimizer_name"] ...
    else:
        raise ValueError("Invalid optimizer name")