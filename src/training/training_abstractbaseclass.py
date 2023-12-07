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
from src.training.earlystopping import EarlyStopping
import matplotlib.pyplot as plt
from time import time
import torch.multiprocessing as mp

# defines an abstract base class for training
class ABCTrainingModule(ABC):
    def __init__(self, model, optimizer, config) -> None:
        params = config["training"]
        self.tag = config["options"]["saving_tag"]
        self.model = model
        self.optimizer = optimizer
        self.batch_size = params.get("batch_size", 16)
        self.epoch = 0
        self.use_wandb = config["options"]["use_wandb"]
        self.make_gradients_plot = config["options"]["make_gradients_plot"]

        self.total_seq_length = params["total_seq_length"]
        self.n_intervals = int((params["total_seq_length"] - config["options"]["length_skipped_in_data"]) / params["seq_length"])
        self.seq_length = params["seq_length"]
        assert((params["total_seq_length"] - config["options"]["length_skipped_in_data"]) % self.n_intervals == 0)

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
        self.coherencies_trial, self.conditionIds, self.train_dataset, self.test_dataset, self.val_dataset = dataset_creator(config)

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )

        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
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

        # save coherency + condition IDs
        self.save_history_coherency_conditionIds()

        print("Using device:", self.device)
        self.model.to(self.device)

    def fit(self, num_epochs: int = 100):
        self.model.train()
        train_loss_history = []
        es = EarlyStopping()
        # multiprocessing
        #self.model.share_memory()
        #processes = []
        #
        for cur_epoch in (pbar_epoch := tqdm(range(num_epochs))):
            self.epoch = cur_epoch
            self.running_loss = 0.0
            gradients = [] # for gradient clipping

            if(self.make_weights_histograms):
                self.create_weights_histogram(cur_epoch, num_epochs)
            if(self.make_hidden_state_plot):
                self.create_hidden_state_plot(cur_epoch, num_epochs)

            # splitting up training https://medium.com/mindboard/training-recurrent-neural-networks-on-long-sequences-b7a3f2079d49

            #t3 = time()
            # multiprocessing
            """
            self.model.train()
            q = mp.Queue()
            for rank in range(self.num_processes):
                p = mp.Process(target=self.train_multi, args=(gradients, self.n_intervals, self.device, self.train_dataloader, q, self.seq_length, self.step))
                p.start()
                processes.append(p)
                self.running_loss += q.get()
            for p in processes:
                p.join()
            self.running_loss /= self.num_processes
            """
            #
            self.model.train()
            self.train(gradients, self.n_intervals,self.device, self.train_dataloader, self.step, (self.training_help>cur_epoch) )

            self.model.eval()
            self.val_loss = 0
            self.train(gradients, self.n_intervals, self.device, self.val_dataloader, self.step, (self.training_help>cur_epoch) )

            #t4 = time()
            #print("-------- time of one iteration: ", t4-t3)

            if self.use_wandb:
                wandb.log({"loss": self.running_loss,
                           "val_loss": self.val_loss})
            if(cur_epoch % 10 == 0):
                train_loss_history.append(self.running_loss)

            if es(self.model, self.val_loss):
                if(self.make_weights_histograms):
                    self.create_weights_histogram(cur_epoch, num_epochs, force_create=True)
                if(self.make_hidden_state_plot):
                    self.create_hidden_state_plot(cur_epoch, num_epochs, force_create=True)
                break
            pbar_description = f"Epoch[{cur_epoch + 1}/{num_epochs}], Loss: {self.running_loss / len(self.train_dataloader):.4f}, Val_L: {self.val_loss / len(self.val_dataloader):.4f} , EStop:[{es.status}]"

            pbar_epoch.set_description(pbar_description)


        self.save_train_loss_hist(train_loss_history)

        self.save_model(self.tag) 
    
    def train(self, gradients, n_intervals, device, t_loader, f, train_help=False):
        for i, (inputs, targets) in enumerate(t_loader):
            h_1 = None
            trial_loss = torch.zeros((1))
            for j in range(n_intervals):
                do_backprop = j == (n_intervals-1)
                if train_help: # TRAINING help for faster initial training
                    do_backprop = True
                inter = j*self.seq_length
                val = (j+1)*self.seq_length if j != (self.n_intervals-1) else None

                partial_in, partial_tar = inputs[:,inter:val,:], targets[:,inter:val,:]
                partial_in, partial_tar = partial_in.to(device), partial_tar.to(device)
                
                out, loss, h_1, trial_loss = f(partial_in, partial_tar, train_help, h_1, gradients=gradients, backprop=do_backprop, trial_loss=trial_loss)

                h_1 = h_1.detach()

                if self.model.training:
                    self.running_loss += loss
                else :
                    self.val_loss += loss

    def train_multi(self, gradients, n_intervals, device, t_loader, q, seq_length, f, train_help=False):
        for i, (inputs, targets) in enumerate(t_loader):
            h_1 = None
            trial_loss = torch.zeros((1))
            for j in range(n_intervals):
                do_backprop = j == (n_intervals-1)
                if train_help: # TRAINING help for faster initial training
                    do_backprop = True
                inter = j*seq_length
                val = (j+1)*seq_length if j != (n_intervals-1) else None

                partial_in, partial_tar = inputs[:,inter:val,:], targets[:,inter:val,:]
                partial_in, partial_tar = partial_in.to(device), partial_tar.to(device)
                
                out, loss, h_1, trial_loss = f(partial_in, partial_tar, train_help, h_1, gradients=gradients, backprop=do_backprop, trial_loss=trial_loss)

                h_1 = h_1.detach()

                q.put(trial_loss.item())
          
    def test(self, model_tag):
        #"""Test the model and save the results"""
        self.model.load_state_dict(
            torch.load(self.output_path / f"{model_tag}_model.pt")
        )
        print("---- testing with model " + model_tag + " ----")
        self.model.eval()
        running_test_loss = 0.0
        test_predictions = []
        test_labels = []

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.test_dataloader):
                h_1 = None
                whole_seq_out = None

                for j in range(self.n_intervals):
                    inter = j*self.seq_length
                    val = (j+1)*self.seq_length if j != (self.n_intervals-1) else None
                    partial_in, partial_tar = inputs[:,inter:val,:], targets[:,inter:val,:]
                    partial_in, partial_tar = partial_in.to(self.device), partial_tar.to(self.device)

                    out, loss, h_1, _ = self.step(partial_in, partial_tar, h_1, eval=True)

                    h_1 = h_1.detach()
                    
                    running_test_loss += loss
                    whole_seq_out = out if whole_seq_out is None else torch.cat( (whole_seq_out, out), 1)
                    
                    if (i == 0) and (j == (self.n_intervals -1)): 
                        self.print_samples(whole_seq_out, targets)
                whole_seq_out = whole_seq_out.squeeze(2)
                targets = targets.squeeze(2)
                test_predictions.append(whole_seq_out)
                test_labels.append(targets)
                
        test_metrics = self.compute_metrics(
            test_predictions, test_labels
        )

        print(f"Model {model_tag}")
        print(test_metrics)

    def print_samples(self, out, targets):
        print("--------------- Example Test ---------------")  
        for i in range(len(out[0,:,0])):
            if(i % 10 == 0):
                print(i, "\t", round(out[0,i,0].item(), 4), "\t", round(targets[0,i,0].item(), 4))
        print("-------------------------------------------")

    def step(self, inputs, labels, train_help=False, h_1=None, gradients=None, eval=False, backprop=True, trial_loss=None):
        """Returns loss"""
        
        out, loss, h_1 = self.compute_loss(inputs, labels, h_1)
        step_loss = loss.item()
        
        if not eval:
            self.optimizer.zero_grad()
            trial_loss += loss
            if backprop:
                if train_help:
                    loss.backward()
                else:
                    trial_loss.backward()
            
            if self.gradient_clipping and backprop:
                self.plot_gradients(gradients)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.) # changed to 2 clip
            self.optimizer.step()
        return out, step_loss, h_1, trial_loss   
    
    def get_activity_and_data_for_op_dimension(self, new_weights=None, revert_weights_at_end=False):
        n_total_trials = self.coherencies_trial.shape[1]
        n_train_trials = round(n_total_trials * 0.75)
        n_test_trials = round(n_total_trials * 0.8)

        if new_weights:
            n_Wru_v_old, n_Wrr_n_old, m_Wzr_n_old = self.model.get_all_weight_matrices()
            self.model.set_weight_matrices(new_weights[0], new_weights[1], new_weights[2])

        all_activities = torch.empty((0))
        for i, (inputs, targets) in enumerate(self.val_dataloader):
            h_1 = None
            activity = torch.empty((0))
            
            for j in range(self.n_intervals):
                inter = j*self.seq_length
                val = (j+1)*self.seq_length if j != (self.n_intervals-1) else None

                partial_in = inputs[:,inter:val,:]
                partial_in = partial_in.to(self.device)
                
                out, h_1,  = self.model.get_activity(partial_in, h_1)

                activity = torch.cat((activity, out), dim=1)
            all_activities = torch.cat((all_activities, activity), dim=0)

        if revert_weights_at_end:
            self.model.set_weight_matrices(n_Wru_v_old, n_Wrr_n_old, m_Wzr_n_old)

        return all_activities, self.coherencies_trial[:,n_test_trials:], self.conditionIds[:,n_test_trials:]

    def run_one_forwardPass(self, n_Wru_v, n_Wrr_n, m_Wzr_n, h1, noise_sigma, inputs):
        forwardPass = {};

        n_Wru_v_old, n_Wrr_n_old, m_Wzr_n_old = self.model.get_all_weight_matrices()
        self.model.set_weight_matrices(n_Wru_v, n_Wrr_n, m_Wzr_n)
        
        original_hidden_noise = self.model.get_hidden_noise()
        self.model.set_hidden_noise(noise_sigma)

        out, h_1 = self.model.get_activity(inputs, h1)

        forwardPass['n_x_t'], forwardPass['n_x0_1'] = out.detach().numpy(), torch.zeros(out.shape).detach().numpy()

        # revert
        self.model.set_hidden_noise(original_hidden_noise)
        self.model.set_weight_matrices(n_Wru_v_old, n_Wrr_n_old, m_Wzr_n_old)
        
        return forwardPass
    
    def run_one_forwardPass_on_val_set(self, n_Wru_v, n_Wrr_n, m_Wzr_n, noise_sigma=0):
        forwardPass = {};

        # change model
        n_Wru_v_old, n_Wrr_n_old, m_Wzr_n_old = self.model.get_all_weight_matrices()
        self.model.set_weight_matrices(n_Wru_v, n_Wrr_n, m_Wzr_n)
        
        original_hidden_noise = self.model.get_hidden_noise()
        self.model.set_hidden_noise(noise_sigma)

        # get pred
        all_activities = torch.empty((0))
        all_m_z_t = torch.empty((0))
        all_targets = torch.empty((0))

        for i, (inputs, targets) in enumerate(self.val_dataloader):
            h_1_act = None
            h_1_m_z = None
            activity = torch.empty((0))
            m_z_t = torch.empty((0))
            
            for j in range(self.n_intervals):
                inter = j*self.seq_length
                val = (j+1)*self.seq_length if j != (self.n_intervals-1) else None

                partial_in = inputs[:,inter:val,:]
                partial_in = partial_in.to(self.device)
                
                out, h_1_m_z  = self.model(partial_in, h_1_m_z)
                m_z_t = torch.cat((m_z_t, out), dim=1)

                out, h_1_act = self.model.get_activity(partial_in, h_1_act)
                activity = torch.cat((activity, out), dim=1)
                
            all_activities = torch.cat((all_activities, activity), dim=0)
            all_m_z_t = torch.cat((all_m_z_t, m_z_t), dim=0)
            all_targets = torch.cat((all_targets, targets), dim=0)
        
        forwardPass["n_x_t"] = all_activities.permute((2,1,0)).detach().numpy()
        forwardPass["m_z_t"] = all_m_z_t.permute((2,1,0)).detach().numpy()
        all_targets = all_targets.permute((2,1,0)).detach().numpy()

        # revert changes
        self.model.set_hidden_noise(original_hidden_noise)
        self.model.set_weight_matrices(n_Wru_v_old, n_Wrr_n_old, m_Wzr_n_old)

        return forwardPass, all_targets

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

    def save_train_loss_hist(self, train_loss_history):
        # Save histories as numpy arrays
        np.save(
            self.output_path / "train_loss_history.npy", np.array(train_loss_history)
        )

        np.savetxt(
            self.output_path / "train_loss_history.txt", np.array(train_loss_history)
        )

    def save_history_coherency_conditionIds(self):
        np.save(
            self.output_path / "coherencies_trial.npy", np.array(self.coherencies_trial)
        )

        np.save(
            self.output_path / "conditionIds.npy", np.array(self.conditionIds)
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

        for _, (inputs, targets) in enumerate(self.test_dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            out, _, _, _ = self.step(inputs, targets, eval=True)

            whole_pred = out if whole_pred is None else torch.cat( (whole_pred, out), 0)
            whole_tar = targets if whole_tar is None else torch.cat( (whole_tar, targets), 0)

        for _, (inputs, targets) in enumerate(self.val_dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            out, _, _, _ = self.step(inputs, targets, eval=True)

            whole_pred = out if whole_pred is None else torch.cat( (whole_pred, out), 0)
            whole_tar = targets if whole_tar is None else torch.cat( (whole_tar, targets), 0)

        return whole_pred, whole_tar
    
    def create_weights_histogram(self, cur_epoch, num_epochs, force_create=False):
        if(force_create or (1. * (cur_epoch-1) / num_epochs) >= (1. / (self.n_weight_histograms-1) * self.cur_weight_hist)):
            self.cur_weight_hist += 1
            w_in, w_rr, w_out = self.model.get_weight_matrices()
            self.save_histogram(w_in, "w_in", cur_epoch, num_epochs)
            self.save_histogram(w_rr, "w_rr", cur_epoch, num_epochs)
            self.save_histogram(w_out, "w_out", cur_epoch, num_epochs)
       
    def save_histogram(self, weights, name, cur_epoch, num_epochs):

        plt.hist(weights, bins=50) 
        plt.title("%s Histogram at epoch %0.0f of %0.0f total epochs"%(name, cur_epoch, num_epochs))
        plt.xlabel("weight size")
        plt.ylabel("Frequency")  
        plt.xlim(min(-1,weights.min()), max(1,weights.max())) 
        filepath_name = "histogram_%s_%0.0f.png"%(name,cur_epoch)
        plt.savefig(self.output_path_plots / filepath_name)
        if self.use_wandb:
            plot_name = "histogram_%s_%0.0f"%(name,cur_epoch)
            im = plt.imread(self.output_path_plots  / filepath_name)
            wandb.log({"img_%s"%name: [wandb.Image(im, caption=plot_name)]})
        plt.close()
        plt.show()

    def create_hidden_state_plot(self, cur_epoch, num_epochs, force_create=False):
        if(force_create or (1. * (cur_epoch) / num_epochs >= 1. / (self.n_weight_histograms-1) * self.cur_hidden_state_hist)):
            self.cur_hidden_state_hist += 1
            start_h_1_container = torch.empty(0)
            mid_h_1_container = torch.empty(0)
            end_h_1_container = torch.empty(0)
            for i, (inputs, _) in enumerate(self.train_dataloader):
                all_h_1, _ = self.model.get_activity(inputs, h_0=None)
                mid_point = int(len(all_h_1[0,:,0]) / 2)
                start_h_1 = all_h_1[:,0,:].reshape(-1)
                mid_h_1 = all_h_1[:,mid_point,:].reshape(-1)
                end_h_1 = all_h_1[:,-1,:].reshape(-1)

                start_h_1_container = torch.concat((start_h_1_container,start_h_1))
                mid_h_1_container = torch.concat((mid_h_1_container,mid_h_1))
                end_h_1_container = torch.concat((end_h_1_container,end_h_1))

            self.save_histogram(start_h_1_container.reshape(-1).detach().numpy(), "t_start_hidden_state", cur_epoch + 1, num_epochs)
            self.save_histogram(mid_h_1_container.reshape(-1).detach().numpy(), "t_mid_hidden_state", cur_epoch + 1, num_epochs)
            self.save_histogram(end_h_1_container.reshape(-1).detach().numpy(), "t_end_hidden_state", cur_epoch + 1, num_epochs)

    @abstractmethod
    def compute_loss(self, inputs, labels):
        """Returns loss"""
        pass

    @abstractmethod
    def compute_metrics(self, model_predictions, labels):
        pass