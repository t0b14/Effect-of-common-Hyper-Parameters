import numpy as np
import torch.optim as optim


def optimizer_factory(weights, params):
    if params["optimizer_name"] == "lbfgs":
        return optim.LBFGS(
            weights,
            lr=float(params["lr"]),
            max_iter=params["max_iter"],
            max_eval=params["max_eval"],
            history_size=params["history_size"],
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )
    elif params["optimizer_name"] == "adam":
        return optim.Adam(
            weights,
            lr=float(params["lr"]),
            betas=params.get("betas", (0.9, 0.999)),
            eps=params.get("eps", 1e-7),
            weight_decay=params.get("weight_decay", 0.0),
            amsgrad=params.get("amsgrad", False),
        )
    else:
        raise ValueError("Invalid optimizer name")