from src.model import model_factory
from src.optimizer import optimizer_factory
from src.training.classification_training_module import ClassificationTrainingModule
from src.training.segmentation_unet_training_module import SegmentationUNetTrainingModule
from src.training.segmentation_training_module import SegmentationTrainingModule
from src.training.vae_training_module import VAETrainingModule


def run(config):
    model = model_factory(config["model"])

    optimizer = optimizer_factory(model.parameters(), config["optimizer"])

    # Setup training module
    tm = None
    if config.get("mode") == "vae":
        tm = VAETrainingModule(model, optimizer, config["training"])
    elif config.get("mode") == "classification":
        tm = ClassificationTrainingModule(
            model, optimizer, config["training"], config["model"]["out_dim"]
        )
    elif config.get("mode") == "segmentation_unet":
        tm = SegmentationUNetTrainingModule(
            model, optimizer, config["training"], config["model"]["out_dim"]
        )
    elif config.get("mode") == "segmentation":
        tm = SegmentationTrainingModule(
            model, optimizer, config["training"], config["model"]["out_dim"]
        )
    else:
        raise ValueError("Invalid mode")

    # Test all trained models
    model_tags = tm.fit(num_epochs=config["training"]["n_epochs"])

    for tag in model_tags:
        tm.test(tag)

    print(f"Number of parameters: {tm.model.num_params}")