
from pathlib import Path
import torch
import numpy as np
from src.constants import MODEL_DIR

def plot_h(tm):

    _, coh_trial, condIds = tm.get_output_paths()
    coherency = np.load(coh_trial)
    conditionIds = np.load(condIds)



    cond_motion_ind = (conditionIds == 1)[0,:].reshape((-1))
    cond_color_ind = conditionIds == 2
    print(coherency[0,coh_trial])
    print(cond_motion_ind.shape)
    print(cond_color_ind)
    #print(coherency.shape) (2,60) 
    #print(conditionIds.shape) (1,60)
    the_model = tm.get_model("last")
    #print(coherency[0, (conditionIds[0,:] == 1)])
    #print("------------------")
    #print(conditionIds)
