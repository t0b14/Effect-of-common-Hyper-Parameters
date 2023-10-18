from src.network import rnn

def model_creator(params):

    if params["network_name"] == "rnn":
        return rnn(
            in_dim=params["in_dim"],
            out_dim=params["out_dim"],
            hidden_dims=params["hidden_dims"],
        )
    # if params["network_name"] ...
    else:
        raise ValueError("Invalid network name")