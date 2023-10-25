
from pathlib import Path
import torch
import numpy as np
from src.constants import MODEL_DIR
import matplotlib.pyplot as plt

def plot_h(tm, params, tag=None):
    _, coh_trial, condIds = tm.get_output_paths()
    coherency = np.load(coh_trial)
    conditionIds = np.load(condIds)

    cond_motion_ind = (conditionIds == 1)[0,:]
    cond_color_ind = (conditionIds == 2)[0,:]

    
    if tag is None:
        path = tm.get_model("last")
    else:
        path = tm.get_model(tag)

    pred, tar = tm.output_whole_dataset()

    direction = pred[:,-10:,:].mean(dim=1).reshape(-1)

    chose_right = direction > 0.

    possible_move_vals = np.sort(np.unique(coherency[0,:]))
    possible_col_vals = np.sort(np.unique(coherency[1,:]))

    #Plot move choices
    total_move_val_counter = np.zeros(possible_move_vals.shape)
    pred_move_val_counter = np.zeros(possible_move_vals.shape)
    
    for i, thinks_right in enumerate(chose_right):
        if cond_motion_ind[i]:
            move_val = coherency[0,i] 
            j = np.where(possible_move_vals == move_val)[0]
            pred_move_val_counter[j] += int(thinks_right)
            total_move_val_counter[j] += 1.
    
    plt.plot(possible_move_vals, pred_move_val_counter/total_move_val_counter, linewidth=5)
    plt.ylabel("p chose right based on " + str(int(total_move_val_counter.mean())))
    plt.xlabel("coherency values")
    plt.ylim((0,1))
    plt.title("move plot")
    plt.savefig(path / 'move.pdf')
    if params["show_plot"] == 0:
        plt.close()
    plt.show()

    # Plot COLOR choices
    total_col_val_counter = np.zeros(possible_col_vals.shape)
    pred_col_val_counter = np.zeros(possible_col_vals.shape)
    
    for i, thinks_right in enumerate(chose_right):
        if cond_color_ind[i]:
            col_val = coherency[1,i] 
            j = np.where(possible_col_vals == col_val)[0]
            pred_col_val_counter[j] += int(thinks_right)
            total_col_val_counter[j] += 1.
    
    plt.plot(possible_move_vals, pred_col_val_counter/total_col_val_counter, linewidth=5)
    plt.ylabel("p chose right based on " + str(int(total_col_val_counter.mean())))
    plt.xlabel("coherency values")
    plt.ylim((0,1))
    plt.title("col plot")
    plt.savefig(path / 'color.pdf')
    if params["show_plot"] == 0:
        plt.close()
    plt.show()


