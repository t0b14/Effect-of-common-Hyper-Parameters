
from pathlib import Path
import torch
import numpy as np
from src.constants import MODEL_DIR

def plot_h(tm):

    _, coh_trial, condIds = tm.get_output_paths()
    coherency = np.load(coh_trial)
    conditionIds = np.load(condIds)

    #print(coherency.shape) (2,60) 
    #print(conditionIds.shape) (1,60)

    #print(coherency[0, (conditionIds[0,:] == 1)])
    #print("------------------")
    #print(conditionIds)
