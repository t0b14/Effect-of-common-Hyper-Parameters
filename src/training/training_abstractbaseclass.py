from abc import ABC, abstractmethod
from pathlib import Path
import json
from tqdm import tqdm

import torch
import numpy as np

from src.training.data import dataset_creator
from src.constants import MODEL_DIR

# defines an abstract base class for training
class ABCTrainingModule(ABC):
    def __init__(self, model, optimizer, params) -> None:
        self.model = model
        self.optimizer = optimizer
        self.batch_size = params.get("batch_size", 16)
        self.epoch = 0
        
        # Load dataset
        # (input_shape, n_timesteps, n_trials)
        self.coherencies_trial, self.conditionIds, self.dataset, self.test_dataset = dataset_creator(params)

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset, [0.8, 0.2]
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )

        # Setup output directory
        self.output_path = Path(params["output_path"])
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        print("Using device:", self.device)
        self.model.to(self.device)


    def fit(self, num_epochs: int = 100):
        #TODO
        best_val_loss = float("inf")
        train_loss_history = []
        val_loss_history = []
        val_metrics_history = []
        for cur_epoch in (pbar_epoch := tqdm(range(num_epochs))):
            self.epoch = cur_epoch
            running_loss = 0.0
            for _, (images, labels) in enumerate(self.train_dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                out, loss = self.step(images, labels)
                running_loss += loss
                train_loss_history.append(loss)

            running_val_loss = 0.0
            val_predictions = []
            val_labels = []
            with torch.no_grad():
                for _, (images, labels) in enumerate(self.val_dataloader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    out, loss = self.step(images, labels, eval=True)
                    running_val_loss += loss
                    val_predictions.append(out)
                    val_labels.append(labels)

                val_loss_history.append(running_val_loss)
                self.last_test_image_batch = images

            # Show metrics in pbar
            pbar_description = f"Epoch[{cur_epoch + 1}/{num_epochs}], Loss: {running_loss / len(self.train_dataloader):.4f}"
            val_metrics = self.compute_metrics(
                torch.cat(val_predictions, 0), torch.cat(val_labels, 0)
            )
            val_metrics_history.append(val_metrics)
            for k, v in val_metrics.items():
                pbar_description += f", Val {k}: {v:.4f}"
            pbar_epoch.set_description(pbar_description)

            # Save best models, hack for reducing io
            if running_val_loss < best_val_loss:
                best_val_loss = running_val_loss
                self.save_model(f"best_val")

            if cur_epoch % 10 == 0 and cur_epoch > 0:
                self.test(f"best_val")

            for k, v in val_metrics.items():
                self.save_model(f"best_val_{k.replace(' ', '_')}")

        # Save histories as numpy arrays
        np.save(
            self.output_path / "train_loss_history.npy", np.array(train_loss_history)
        )
        np.save(self.output_path / "val_loss_history.npy", np.array(val_loss_history))
        np.save(
            self.output_path / "val_metrics_history.npy", np.array(val_metrics_history)
        )

        self.save_model("last")
        return ["last", "best_val"] + [
            f"best_val_{k.replace(' ', '_')}" for k, _ in val_metrics.items()
        ]

    def test(self, model_tag):
        """Test the model and save the results"""
        self.model.load_state_dict(
            torch.load(self.output_path / f"{model_tag}_model.pt")
        )
        self.model.eval()
        running_test_loss = 0.0
        test_predictions = []
        test_lables = []

        with torch.no_grad():
            for _, (images, labels) in enumerate(self.test_dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                out, loss = self.step(images, labels, eval=True)
                running_test_loss += loss
                test_predictions.append(out)
                test_lables.append(labels)

            self.last_test_image_batch = images

        test_metrics = self.compute_metrics(
            torch.cat(test_predictions, 0), torch.cat(test_lables, 0)
        )

        # Save metrics
        with open(self.output_path / f"{model_tag}_test_metrics.json", "w+") as file:
            json.dump(test_metrics, file)

        print(f"Model {model_tag}")
        print(test_metrics)

    def step(self, images, labels, eval=False):
        """Returns loss"""
        out, loss = self.compute_loss(images, labels)
        step_loss = loss.item()

        if not eval:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return out, step_loss

    def save_model(self, tag: str = "last"):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.output_path / f"{tag}_model.pt")

    @abstractmethod
    def compute_loss(self, inputs, labels):
        """Returns loss"""
        pass

    @abstractmethod
    def compute_metrics(self, model_predictions, labels):
        """Returns a dictionary of metrics, the key will be used as the display name in the progress bar"""
        pass