from abc import ABC, abstractmethod
from pathlib import Path
import json
from tqdm import tqdm
import math
import torch
import numpy as np
import sys
from src.training.data import dataset_creator
from src.constants import MODEL_DIR

# defines an abstract base class for training
class ABCTrainingModule(ABC):
    def __init__(self, model, optimizer, params) -> None:
        self.model = model
        self.optimizer = optimizer
        self.batch_size = params.get("batch_size", 16)
        self.epoch = 0

        self.total_seq_length = params["total_seq_length"]
        self.n_intervals = int(params["total_seq_length"] / params["seq_length"])
        assert(self.total_seq_length % self.n_intervals == 0)
        
        self.hidden_dims = params["hidden_dims"]

        # Load dataset
        # (input_shape, n_timesteps, n_trials)
        self.coherencies_trial, self.conditionIds, self.train_dataset, self.test_dataset = dataset_creator(params)

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )
        # TODO validation data set (later)

        # Setup output directory
        self.output_path = Path(params["output_path"])
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        print("Using device:", self.device)
        self.model.to(self.device)


    def fit(self, num_epochs: int = 100):
        self.model.train()
        train_loss_history = []
        for cur_epoch in (pbar_epoch := tqdm(range(num_epochs))):
            self.epoch = cur_epoch
            running_loss = 0.0
            
            # splitting up training https://medium.com/mindboard/training-recurrent-neural-networks-on-long-sequences-b7a3f2079d49
            for i, (inputs, targets) in enumerate(self.train_dataloader):
                h_1 = None

                for j in range(self.n_intervals):
                    inter = j*self.n_intervals
                    val = (j+1)*self.n_intervals if j != (self.n_intervals-1) else None
                    partial_in, partial_tar = inputs[:,inter:val,:], targets[:,inter:val,:]
                    partial_in, partial_tar = partial_in.to(self.device), partial_tar.to(self.device)

                    out, loss, h_1 = self.step(partial_in, partial_tar, h_1)
                    h_1 = h_1.detach()

                    running_loss += loss
            
            if(cur_epoch % 10 == 0):
                train_loss_history.append(loss)
                
            if(cur_epoch % num_epochs == num_epochs-1):  
                print("--------------- Train ---------------")  
                for i in range(len(out[0,:,0])):
                    if(i % 100 == 0):
                        print(i, "\t", round(out[0,i,0].item(), 4), "\t", round(targets[0,i,0].item(),4))
                print("-------------------------------------")

            pbar_description = f"Epoch[{cur_epoch + 1}/{num_epochs}], Loss: {running_loss / len(self.train_dataloader):.4f}"
            pbar_epoch.set_description(pbar_description)

        self.save_history_coherency_conditionIds(train_loss_history)

        self.save_model("last") # TODO expand saving capabilities

    def test(self, model_tag="last"):
        """Test the model and save the results"""
        self.model.load_state_dict(
            torch.load(self.output_path / f"{model_tag}_model.pt")
        )
        self.model.eval()
        running_test_loss = 0.0
        test_predictions = []
        test_labels = []

        with torch.no_grad():
            for _, (inputs, targets) in enumerate(self.test_dataloader):
                h_1 = None
                whole_seq_out = None

                for j in range(self.n_intervals):
                    inter = j*self.n_intervals
                    val = (j+1)*self.n_intervals if j != (self.n_intervals-1) else None
                    partial_in, partial_tar = inputs[:,inter:val,:], targets[:,inter:val,:]
                    partial_in, partial_tar = partial_in.to(self.device), partial_tar.to(self.device)

                    out, loss, h_1 = self.step(partial_in, partial_tar, h_1, eval=True)
                    h_1 = h_1.detach()
            
                    running_test_loss += loss

                    whole_seq_out = out if whole_seq_out is None else torch.cat( (whole_seq_out, out), 1)

                test_predictions.append(whole_seq_out)
                test_labels.append(targets)

        print("--------------- Example Test ---------------")  
        for i in range(len(out[0,:,0])):
            if(i % 100 == 0):
                print(i, "\t", round(out[0,i,0].item(), 4), "\t", round(targets[0,i,0].item(), 4))
        print("-------------------------------------")
        
        test_metrics = self.compute_metrics(
            test_predictions, test_labels
        )

        print(f"Model {model_tag}")
        print(test_metrics)

    def step(self, inputs, labels, h_1, eval=False):
        """Returns loss"""
        out, loss, h_1 = self.compute_loss(inputs, labels, h_1)
        step_loss = loss.item()

        if not eval:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            self.optimizer.step()

        return out, step_loss, h_1

    def save_model(self, tag: str = "last"):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.output_path / f"{tag}_model.pt")
        
    def save_history_coherency_conditionIds(self, train_loss_history):
        # Save histories as numpy arrays
        np.save(
            self.output_path / "train_loss_history.npy", np.array(train_loss_history)
        )

        np.save(
            self.output_path / "coherencies_trial.npy", np.array(self.coherencies_trial)
        )

        np.save(
            self.output_path / "conditionIds.npy", np.array(self.conditionIds)
        )

        np.savetxt(
            self.output_path / "train_loss_history.txt", np.array(train_loss_history)
        )

        np.savetxt(
            self.output_path / "coherencies_trial.txt", np.array(self.coherencies_trial)
        )

        np.savetxt(
            self.output_path / "conditionIds.txt", np.array(self.coherencies_trial)
        )

    def get_output_paths(self):
        return (self.output_path / "train_loss_history.npy", self.output_path / "coherencies_trial.npy", self.output_path / "conditionIds.npy")
    
    def get_model(self, model_tag):
        self.model.load_state_dict(
            torch.load(self.output_path / f"{model_tag}_model.pt")
        )

    @abstractmethod
    def compute_loss(self, inputs, labels):
        """Returns loss"""
        pass

    @abstractmethod
    def compute_metrics(self, model_predictions, labels):
        pass