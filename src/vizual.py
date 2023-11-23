
from pathlib import Path
import torch
import wandb
import numpy as np
from src.constants import MODEL_DIR
import matplotlib.pyplot as plt

# coherency: value out of the possible coherency values
# cond_ind: bool stating wether the current context should be used. e.g. is move or is color
# chose_right: did the model predict a positive value (e.g. moved right)
def calculate(coherency, chose_right, cond_ind):
    # array of all the possble coherency values
    possible_vals = np.sort(np.unique(coherency))
    #empty arrays to plot choices
    total_move_val_counter = np.zeros(possible_vals.shape)
    pred_move_val_counter = np.zeros(possible_vals.shape)
    
    for i, thinks_right in enumerate(chose_right):
        if cond_ind[i]:
            move_val = coherency[i] 
            # index of coherency value for current trial
            j = np.where(possible_vals == move_val)[0] 
            pred_move_val_counter[j] += int(thinks_right.item())
            total_move_val_counter[j] += 1.

    return total_move_val_counter, pred_move_val_counter, possible_vals

def custom_plot(possible_vals, pred_val_counter, total_val_counter, path, params, name):
    percent_correct = pred_val_counter/total_val_counter
    percent_correct[np.isnan(percent_correct)] = 0
    plt.plot(possible_vals, percent_correct, '--bo', markersize=15, linewidth=5)
    plt.ylabel("p chose right based on " + str(int(total_val_counter.mean())))
    plt.xlabel("coherency values")
    plt.ylim((0,1))
    plt.title(name +" plot")
    plt.savefig(path / name)
    if params["use_wandb"]:
        wandb.log({name: plt})
    if params["show_plot"] == 0:
        plt.close()
    plt.show()


def plot_h(tm, params, tag):
    # load coherency and conditionIds from files
    _, coh_trial, condIds = tm.get_output_paths()
    coherency = np.load(coh_trial)
    coherency = coherency[:,int(len(coherency[0,:])*0.75):]

    # test for move or color
    conditionIds = np.load(condIds)
    conditionIds = conditionIds[:,int(len(conditionIds[0,:])*0.75):]

    cond_motion_ind = (conditionIds == 1)[0,:]
    cond_col_ind = (conditionIds == 2)[0,:]

    path = tm.get_model(tag)

    pred, tar = tm.output_whole_dataset() # val and test data set predictions
    
    direction = pred[:,-10:,:].median(dim=1)[0].reshape(-1).detach().numpy()
    
    chose_right = direction > 0.

    # replace with chose_right for correct result (bugfixing)
    correct_direction_right = (tar[:,-10:,:].median(dim=1)[0].reshape(-1) > 0.).numpy()

    #print("--- ", (correct_direction_right == chose_right).to(torch.float64).mean().item(), " ", len((correct_direction_right == chose_right)))
    #print("---", ((coherency[0,np.array(cond_motion_ind)] > 0.) == correct_direction_right[np.array(cond_motion_ind)]).mean())
    # plot move  
    total_move_val_counter, pred_move_val_counter, possible_move_vals = calculate(coherency[0,:], chose_right, cond_motion_ind)
    custom_plot(possible_move_vals, pred_move_val_counter, total_move_val_counter, path, params,  "move.png")

    
    # plot color
    total_col_val_counter, pred_col_val_counter, possible_col_vals = calculate(coherency[1,:], chose_right, cond_col_ind)
    custom_plot(possible_col_vals, pred_col_val_counter, total_col_val_counter, path, params, "color.png")


    total_move_val_counter, pred_move_val_counter, possible_move_vals = calculate(coherency[1,:], chose_right, cond_motion_ind)
    custom_plot(possible_move_vals, pred_move_val_counter, total_move_val_counter, path, params, "move2.png")


    total_col_val_counter, pred_col_val_counter, possible_col_vals = calculate(coherency[0,:], chose_right, cond_col_ind)
    custom_plot(possible_col_vals, pred_col_val_counter, total_col_val_counter, path, params, "color2.png")
    

    

 
    




