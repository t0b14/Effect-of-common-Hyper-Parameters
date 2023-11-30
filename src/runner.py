from src.optimizer import optimizer_creator
from src.network.rnn import cRNN
from src.training.training_rnn_ext1 import RNNTrainingModule1
from src.vizual import plot_h
import wandb

def init_wandb(config):
    t_params = config["training"]
    opt_params = config["optimizer"]

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=config["title"],
        
        name= "lenskip_hidden_data " + str(config["options"]["length_skipped_in_data"]) + " "+ str(config["model"]["hidden_noise"]) + " "+ str(config["training"]["with_inputnoise"]),
        #config["title"],"apply_gradient_clipping " + str(config["optimizer"]["apply_gradient_clipping"]),#config["options"]["run_name"], #str(config["training"]["noise_level"]),

        # track hyperparameters and run metadata
        config={
            "dataset_name": t_params["dataset_name"],
            "n_epochs": t_params["n_epochs"],
            "batch_size": t_params["batch_size"],
            "total_seq_length": t_params["total_seq_length"],
            "seq_length": t_params["seq_length"],
            "training_help": t_params["training_help"], # 0 to n_epochs (iterations)
            "hidden_dims": t_params["hidden_dims"],
            "n_trials": t_params["n_trials"],
            "with_inputnoise": t_params["with_inputnoise"],
            "noise_level": t_params["noise_level"], # number from 1 to 10; requires with_inputnoise; 0.15 * noiselevel
            "coherency_intervals": t_params["coherency_intervals"], # "original" or "uniform"
            "optimizer_name": opt_params["optimizer_name"],
            "lr": opt_params["lr"],
            "betas": opt_params["betas"],
            "eps": opt_params["eps"],
            "weight_decay": opt_params["weight_decay"],
            "amsgrad": opt_params["amsgrad"],
            "momentum": opt_params["momentum"],
            "apply_gradient_clipping": opt_params["apply_gradient_clipping"],
            "tau": config["model"]["tau"],
            "length_skipped_in_data": config["options"]["length_skipped_in_data"],
            "hidden_noise": config["model"]["hidden_noise"]
        }
    )
# setup and run 
def run(config):
    """"
    for hidden_noise in [0,0.1]:
        for with_inputnoise in [0,1]:
            config["model"]["hidden_noise"] = hidden_noise
            config["training"]["with_inputnoise"] = with_inputnoise
    """
    for tag in ["one","two","third"]:
        config["options"]["saving_tag"] = tag
        
        params = config["model"]

        if config["options"]["use_wandb"]:
            init_wandb(config)

        model = cRNN(
                    config["model"],
                    input_s=params["in_dim"],
                    output_s=params["out_dim"],
                    hidden_s=params["hidden_dims"],
                    hidden_noise=params["hidden_noise"]
                    )

        optimizer = optimizer_creator(model.parameters(), config["optimizer"])

        tm = RNNTrainingModule1(model, optimizer, config)

        if config["options"]["train_n_test"]:
            #train
            tm.fit(num_epochs=config["training"]["n_epochs"])
            #test
            tm.test(config["options"]["saving_tag"])

        if config["options"]["visualize"]:
            plot_h(tm, config["options"], config["options"]["saving_tag"])

        if config["options"]["use_wandb"]:
            wandb.finish()