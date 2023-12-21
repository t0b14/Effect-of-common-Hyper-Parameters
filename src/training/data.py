import torch
import torch.nn.functional as F

import numpy as np
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import random

from src.constants import INPUT_DIR

# directly copied from https://gitlab.com/neuroinf/operativeDimensions/-/tree/master/code_python/utils/input_generation?ref_type=heads
class InputGeneratorCtxt(object):
    # class to generate input/targets for context-dependent integration
    def __init__(self, params):
        self.n_inputs = 5
        self.n_inputs_actual = 4 # remove layers at the end
        self.n_outputs = 1
        self.nIntegrators = 2
        self.maxBound = 1
        self.minBound = -1
        self.integrationTau = 5 * 0.01
        self.noiseLevel = params["noise_level"]
        assert 1. <= self.noiseLevel <= 10., " 1 <= noiseLevel <= 10" 

        self.burnLength = 0.650
        self.dt = 0.001
        self.tau = 0.01
        self.time = 0.75
        if params["coherency_intervals"] == "original":
            self.all_coherencies = np.array([-0.5, -0.12, -0.03, 0.03, 0.12, -0.5]) * 0.3
        elif params["coherency_intervals"] == "uniform":
            self.all_coherencies = np.linspace(-0.5, 0.5, num=11) * 0.3
        else:
            raise ValueError("Invalid coherency protocol")

    def get_ctxt_dep_integrator_inputOutputDataset(self, n_trials, with_inputnoise):
        # main method to generate input/outputs
        # n_trials: (integer), number of trials per context and per coherency
        # with_inputnoise: (bool), if true: add noise to sensory inputs

        # coherencies_trial = [nIntegrators x N], input coherencies of sensory input 1 and 2 over trials
        # coherencies are shuffled. in particular the different coherencies within a
        # trial are independent.
        # conditionIds = [1 x N], context ID per trial
        # inputs: [2*nIntegrators+1 x T x N], sensory and context input over time and trials
        # targets: [1 x T x N], target output over time and trials

        # noise settings
        input_noise = 0
        if with_inputnoise:
            input_noise = 31.6228 * np.sqrt(self.dt) * (self.noiseLevel * 0.1)
        noiseSigma = input_noise

        # generate inputs/targets
        n_trials_total = n_trials * len(self.all_coherencies) * self.nIntegrators
        nTimesteps = int(self.time / self.dt) + int(self.burnLength / self.dt)
        inputs = np.zeros([self.n_inputs, nTimesteps, n_trials_total])
        targets = np.zeros([self.n_outputs, nTimesteps, n_trials_total])
        conditionIds = np.tile(range(1, self.nIntegrators + 1),
                               [1, int(n_trials_total / self.nIntegrators)])

        # set input coherencies per trial
        set_all_cohs = np.zeros([1, n_trials_total])
        for coh_nr in range(len(self.all_coherencies)):
            idx_start = coh_nr * (self.nIntegrators * n_trials)
            idx_end = (coh_nr + 1) * (self.nIntegrators * n_trials)
            set_all_cohs[0, idx_start:idx_end] = np.ones([1, self.nIntegrators * n_trials])\
                                                 * self.all_coherencies[coh_nr]

        coherencies_trial = np.zeros([self.nIntegrators, n_trials_total])
        for i in range(self.nIntegrators):
            coherencies_trial[i, np.where(conditionIds[0, :] == 1)] = \
            set_all_cohs[0, np.random.choice(range(n_trials_total),
                                             size=n_trials * len(self.all_coherencies),
                                             replace=False)]
            coherencies_trial[i, np.where(conditionIds[0, :] == 2)] = \
            set_all_cohs[0, np.random.choice(range(n_trials_total),
                                             size=n_trials * len(self.all_coherencies),
                                             replace=False)]

        # generate one trial
        for trial_nr in range(n_trials_total):
            inputs[:, :, trial_nr], targets[:, :,trial_nr] = self.generate_one_trial(
                coherencies_trial[0:2, trial_nr], conditionIds[0, trial_nr], noiseSigma)
        

        if self.n_inputs > self.n_inputs_actual:
            inputs = inputs[:self.n_inputs_actual,:,:]

        return coherencies_trial, conditionIds, inputs, targets

    def generate_one_trial(self, all_drifts, conditionId, noiseSigma):
        # method to generate input and targets over time for one trial
        # all_drifts: [n_integrators, 1], input coherency for sensory input 1 and 2
        # conditionId: (int), context of trial (1 or 2)
        # noiseSigma: (float), sigma of input noise

        # inputs: (n_inputs, n_timesteps)
        # targets: (n_outputs, n_timesteps)

        nTimesteps = int(self.time / self.dt) + int(self.burnLength / self.dt)
        inputs = np.zeros([self.n_inputs, nTimesteps])
        targets = np.zeros([self.n_outputs, nTimesteps])

        # set context inputs
        inputs[(self.n_inputs - self.nIntegrators - 2) + conditionId, :] = 1

        # set sensory inputs
        for i in range(self.nIntegrators):
            inputs[i, int(self.burnLength / self.dt):] = all_drifts[i]\
                 + np.random.normal(size=nTimesteps - int(self.burnLength / self.dt)) * noiseSigma

        # set target values
        hit_bound = 0
        for t in range(int(self.burnLength / self.dt), nTimesteps):
            if not hit_bound:
                targets[:, t] = targets[:, t - 1] + (self.dt / self.integrationTau) * inputs[conditionId - 1, t]
                if (targets[:, t] >= self.maxBound):
                    targets[:, t] = self.maxBound
                    hit_bound = 1
                elif (targets[:, t] <= self.minBound):
                    targets[:, t] = self.minBound
                    hit_bound = 1
            else:
                targets[:, t] = targets[:, t - 1]

        return inputs, targets

class CustomDataset(Dataset):
    def __init__(self, inputs, target):
        super().__init__()
        self.inputs = torch.tensor(inputs).to(torch.float32) #torch.nn.functional.normalize(torch.tensor(inputs).to(torch.float32), dim=2)
        self.target = torch.tensor(target).to(torch.float32)

    def __len__(self):
        return len(self.inputs[0,0,:])

    def __getitem__(self, idx):
        trial_input = self.inputs[:,:,idx]
        trial_output = self.target[:,:,idx]

        # trial_input: (inputs, timesteps) -> (seq_length, inputs)
        trial_input = trial_input.permute(1,0)
        trial_output = trial_output.permute(1,0)
        
        return trial_input, trial_output
    
def dataset_creator(config):
    random.seed(1000)
    params = config["training"]
    if params["dataset_name"] == "ctxt":
        n_trials, with_inputnoise = params["n_trials"], params["with_inputnoise"]
        # (2,60) (1,60) (4,1400,60) (1,1400,60)
        [coherencies_trial, conditionIds, inputs, targets] = InputGeneratorCtxt(params).get_ctxt_dep_integrator_inputOutputDataset(n_trials, with_inputnoise)  
        # according to coherencies_trial and conditionIds taking last quintile  
        # as test set is representative of train set
        skip_l = config["options"]["length_skipped_in_data"]
        if skip_l > 0:
            inputs = inputs[:,skip_l:,:]
            targets = targets[:,skip_l:,:]

        n_total_trials = len(inputs[0,0,:]) 
        n_train_trials = round(n_total_trials * 0.75)
        n_test_trials = round(n_total_trials * 0.8)

        #correct_direction_right = (np.mean(targets[:,-10:,:], axis=1) > 0.)
        #cond_motion_indexes = (conditionIds == 1)[0,:]
        
        # print accuracy of taking last 10 median values
        #print("---", ((coherencies_trial[0,np.array(cond_motion_indexes)] > 0.) == correct_direction_right[0,np.array(cond_motion_indexes)]).mean())
        
        
        # shuffle
        """
        indexes = torch.randperm(inputs.shape[2])
        inputs = inputs[:,:,indexes]
        coherencies_trial = coherencies_trial[:,indexes]
        conditionIds = conditionIds[:,indexes]
        targets = targets[:,:,indexes]
        """
        train_inputs, train_targets = inputs[:,:,:n_train_trials], targets[:,:,:n_train_trials]
        test_inputs, test_targets = inputs[:,:,n_train_trials:n_test_trials], targets[:,:,n_train_trials:n_test_trials] 
        val_inputs, val_targets = inputs[:,:,n_test_trials:], targets[:,:,n_test_trials:] 

        train_dataset = CustomDataset(train_inputs, train_targets)
        test_dataset = CustomDataset(test_inputs, test_targets)
        val_dataset = CustomDataset(val_inputs, val_targets)

        return [coherencies_trial, conditionIds, train_dataset, test_dataset, val_dataset]
    else:
        raise ValueError("Invalid dataset name")