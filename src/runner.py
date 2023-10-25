from src.optimizer import optimizer_creator
from src.network.rnn import cRNN
from src.training.training_rnn_ext1 import RNNTrainingModule1
from src.vizual import plot_h

# setup and run 
def run(config):

    params = config["model"]

    model = cRNN(input_s=params["in_dim"],
                output_s=params["out_dim"],
                hidden_s=params["hidden_dims"],)

    optimizer = optimizer_creator(model.parameters(), config["optimizer"])

    tm = RNNTrainingModule1(model, optimizer, config["training"])
    
    if config["options"]["train_n_test"]:
        #train
        tm.fit(num_epochs=config["training"]["n_epochs"])
        #test
        tm.test()
    
    if config["options"]["visualize"]:
        plot_h(tm)