from src.network import rnn

def model_factory(params):

    elif params["network_name"] == "rnn":
        return rnn(
            i_dont_know =params["i_dont_know"],
        )
    else:
        raise ValueError("Invalid network name")