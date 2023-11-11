from abc import ABC, abstractmethod
from pathlib import Path
import json
from tqdm import tqdm
import math
import torch
import numpy as np
import sys
import wandb
from src.training.data import dataset_creator
from src.constants import MODEL_DIR
import matplotlib.pyplot as plt
from time import time
import torch.multiprocessing as mp

# defines an abstract base class for training
class ABCTrainingModule(ABC):
    def __init__(self, model, optimizer, config) -> None:
        params = config["training"]
        self.model = model
        self.optimizer = optimizer
        self.batch_size = params.get("batch_size", 16)
        self.epoch = 0
        self.use_wandb = config["options"]["use_wandb"]
        self.make_gradients_plot = config["options"]["make_gradients_plot"]

        self.total_seq_length = params["total_seq_length"]
        self.n_intervals = int(params["total_seq_length"] / params["seq_length"])
        self.seq_length = params["seq_length"]
        assert(self.total_seq_length % self.n_intervals == 0)

        self.gradient_clipping = config["optimizer"]["apply_gradient_clipping"]
        self.training_help = params["training_help"]
        
        self.hidden_dims = params["hidden_dims"]

        self.clip_percentage = 0
        self.firsthistogram = True
        self.max_grad_norm = 20.

        self.make_weights_histograms = config["options"]["make_weights_histograms"]
        self.n_weight_histograms = 4
        self.cur_weight_hist = 0

        self.make_hidden_state_plot = config["options"]["make_hidden_state_plot"]
        self.n_hidden_state_histograms = 3
        self.cur_hidden_state_hist = 0

        self.num_processes = config["model"]["num_processes"]

        # Load dataset
        # (input_shape, n_timesteps, n_trials)
        self.coherencies_trial, self.conditionIds, self.train_dataset, self.test_dataset = dataset_creator(params)

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )

        # Setup output directory
        self.output_path = Path(params["output_path"])
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Setup output directory
        self.output_path_plots = Path(params["output_path_plots"])
        self.output_path_plots.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        print("Using device:", self.device)
        self.model.to(self.device)

    def fit(self, num_epochs: int = 100):
        self.model.train()
        train_loss_history = []

        # multiprocessing
        self.model.share_memory()
        processes = []
        #
        for cur_epoch in (pbar_epoch := tqdm(range(num_epochs))):
            self.epoch = cur_epoch
            self.running_loss = 0.0
            gradients = [] # for gradient clipping

            seq_l = self.train_help(cur_epoch)

            if(self.make_weights_histograms):
                self.create_weights_histogram(cur_epoch, num_epochs)
            if(self.make_hidden_state_plot):
                self.create_hidden_state_plot(cur_epoch, num_epochs)

            # splitting up training https://medium.com/mindboard/training-recurrent-neural-networks-on-long-sequences-b7a3f2079d49

            #t3 = time()
            # multiprocessing
            """
            q = mp.Queue()
            for rank in range(self.num_processes):
                p = mp.Process(target=self.train, args=(gradients, self.n_intervals,self.device, self.train_dataloader, q, self.step))
                p.start()
                processes.append(p)
                self.running_loss += q.get()
            for p in processes:
                p.join()
            """
            #
            self.train(gradients, seq_l, self.n_intervals,self.device, self.train_dataloader, self.step)
            #t4 = time()
            #print("-------- time of one iteration: ", t4-t3)

            if self.use_wandb:
                wandb.log({"loss": self.running_loss})
            if(cur_epoch % 10 == 0):
                train_loss_history.append(self.running_loss)
                
                
            #if(cur_epoch % num_epochs == num_epochs-1):  
            #    print("--------------- Train ---------------")  
            #    for i in range(len(out[0,:,0])):
            #        if(i % 100 == 0):
            #            print(i, "\t", round(out[0,i,0].item(), 4), "\t", round(targets[0,i,0].item(),4))
            #    print("-------------------------------------")

            pbar_description = f"Epoch[{cur_epoch + 1}/{num_epochs}], Loss: {self.running_loss / len(self.train_dataloader):.4f}"
            pbar_epoch.set_description(pbar_description)

        self.save_history_coherency_conditionIds(train_loss_history)

        self.save_model("last") 
    
    def train(self, gradients, seq_l, n_intervals,device, t_loader, f):
        for i, (inputs, targets) in enumerate(t_loader):
            h_1 = None
            for j in range(n_intervals):
            
                inter = j*seq_l
                val = (j+1)*seq_l if j != (self.n_intervals-1) else None

                partial_in, partial_tar = inputs[:,inter:val,:], targets[:,inter:val,:]
                partial_in, partial_tar = partial_in.to(device), partial_tar.to(device)
                
                out, loss, h_1 = f(partial_in, partial_tar, h_1, gradients=gradients)
                h_1 = h_1.detach()

                #q.put(loss)
                self.running_loss += loss
                # stop after training doesn't improve

                


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
                    inter = j*self.seq_length
                    val = (j+1)*self.seq_length if j != (self.n_intervals-1) else None
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

    def step(self, inputs, labels, h_1=None, gradients=None, eval=False):
        """Returns loss"""
        
        out, loss, h_1 = self.compute_loss(inputs, labels, h_1)
        
        step_loss = loss.item()
        
        if not eval:
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.gradient_clipping:
                self.plot_gradients(gradients)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.) # changed to 2 clip
            self.optimizer.step()
        return out, step_loss, h_1

    def plot_gradients(self, gradients):
        if self.gradient_clipping:
            if not gradients: # is empty
                if self.firsthistogram:
                    for param in self.model.parameters():
                        if param.grad is not None:
                            gradients.append(param.grad.view(-1).detach().cpu().numpy())
                    gradients = np.concatenate(gradients)
                    if self.use_wandb and self.make_gradients_plot:
                        wandb.log({"max_grad_norm": self.max_grad_norm})
                    
                        plt.hist(gradients, bins=100, log=True) 
                        plt.title("Gradient Histogram")
                        plt.xlabel("Gradient Value")
                        plt.ylabel("Frequency (log scale)")  
                        plt.xlim(gradients.min(), gradients.max()) 
                        plt.savefig(self.output_path_plots  / "histogram_gradient.png")
                        if self.use_wandb:
                            plot_name = "histogram_%s"%"gradient"
                            im = plt.imread(self.output_path_plots  / "histogram_gradient.png")
                            wandb.log({"hist_gradient": [wandb.Image(im, caption=plot_name)]})
                        plt.close()
                        plt.show()
                        self.firsthistogram = False

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
        return self.output_path_plots

    def output_whole_dataset(self):
        whole_pred = None
        whole_tar = None

        for _, (inputs, targets) in enumerate(self.train_dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            out, _, _ = self.step(inputs, targets, eval=True)

            whole_pred = out if whole_pred is None else torch.cat( (whole_pred, out), 0)
            whole_tar = targets if whole_tar is None else torch.cat( (whole_tar, targets), 0)

        for _, (inputs, targets) in enumerate(self.test_dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            out, _, _ = self.step(inputs, targets, eval=True)

            whole_pred = out if whole_pred is None else torch.cat( (whole_pred, out), 0)
            whole_tar = targets if whole_tar is None else torch.cat( (whole_tar, targets), 0)

        return whole_pred, whole_tar

    def train_help(self,cur_epoch):
        # Help train initially by gradually increasing the seq_length
        if(self.training_help > cur_epoch):
            seq_l = int(self.seq_length / self.training_help * (cur_epoch+1))
        else:
            seq_l = self.seq_length
        return seq_l
    
    def create_weights_histogram(self, cur_epoch, num_epochs):
        if(1. * (cur_epoch+1) / num_epochs >= 1. / (self.n_weight_histograms-1) * self.cur_weight_hist):
            self.cur_weight_hist += 1
            w_in, w_rr, w_out = self.model.get_weight_matrices()
            self.save_histogram(w_in, "w_in", cur_epoch + 1, num_epochs)
            self.save_histogram(w_rr, "w_rr", cur_epoch + 1, num_epochs)
            self.save_histogram(w_out, "w_out", cur_epoch + 1, num_epochs)
       
    def save_histogram(self, weights, name, cur_epoch, num_epochs):
        plt.hist(weights, bins=50) 
        plt.title("%s Histogram at epoch %0.0f of %0.0f total epochs"%(name, cur_epoch, num_epochs))
        plt.xlabel("weight size")
        plt.ylabel("Frequency")  
        plt.xlim(min(-1,weights.min()), max(1,weights.max())) 
        filepath_name = "histogram_%s_%0.0f.png"%(name,cur_epoch)
        plt.savefig(self.output_path_plots  / filepath_name)
        if self.use_wandb:
            plot_name = "histogram_%s_%0.0f"%(name,cur_epoch)
            im = plt.imread(self.output_path_plots  / filepath_name)
            wandb.log({"img_%s"%name: [wandb.Image(im, caption=plot_name)]})
        plt.close()
        plt.show()

    def create_hidden_state_plot(self, cur_epoch, num_epochs):
        if(1. * (cur_epoch+1) / num_epochs >= 1. / (self.n_weight_histograms-1) * self.cur_hidden_state_hist):
            self.cur_hidden_state_hist += 1
            start_h_1_container = torch.empty(0)
            mid_h_1_container = torch.empty(0)
            end_h_1_container = torch.empty(0)
            for i, (inputs, _) in enumerate(self.train_dataloader):
                all_h_1, _ = self.model.forward(inputs)
                start_h_1 = all_h_1[:,0,:].view(-1)
                mid_h_1 = all_h_1[:,700,:].view(-1)
                end_h_1 = all_h_1[:,-1,:].view(-1)

                start_h_1_container = torch.concat((start_h_1_container,start_h_1))
                mid_h_1_container = torch.concat((mid_h_1_container,mid_h_1))
                end_h_1_container = torch.concat((end_h_1_container,end_h_1))

            self.save_histogram(start_h_1_container.view(-1).detach().numpy(), "t_1_hidden_state", cur_epoch + 1, num_epochs)
            self.save_histogram(mid_h_1_container.view(-1).detach().numpy(), "t_700_hidden_state", cur_epoch + 1, num_epochs)
            self.save_histogram(end_h_1_container.view(-1).detach().numpy(), "t_1400_hidden_state", cur_epoch + 1, num_epochs)

    @abstractmethod
    def compute_loss(self, inputs, labels):
        """Returns loss"""
        pass

    @abstractmethod
    def compute_metrics(self, model_predictions, labels):
        pass