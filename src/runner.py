from src.model import model_factory
from src.optimizer import optimizer_factory
from src.training.training_rnn_ext1 import RNNTrainingModule1


def run(config):
    model = model_factory(config["model"])

    optimizer = optimizer_factory(model.parameters(), config["optimizer"])

    tm = VAETrainingModule(model, optimizer, config["training"])

    # Test all trained models
    model_tags = tm.fit(num_epochs=config["training"]["n_epochs"])

    for tag in model_tags:
        tm.test(tag)

    print(f"Number of parameters: {tm.model.num_params}")