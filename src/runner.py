from src.model import model_creator
from src.optimizer import optimizer_creator
from src.network import rnn

from src.training.training_rnn_ext1 import RNNTrainingModule1


# setup and run 
def run(config):

    params = config["model"]
    model = rnn(in_dim=params["in_dim"],
                out_dim=params["out_dim"],
                hidden_dims=params["hidden_dims"],)

    optimizer = optimizer_creator(model.parameters(), config["optimizer"])

    tm = RNNTrainingModule1(model, optimizer, config["training"])
    
    #train
    model_tags = tm.fit(num_epochs=config["training"]["n_epochs"])

    #test
    for tag in model_tags:
        tm.test(tag)