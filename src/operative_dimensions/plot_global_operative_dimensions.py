from samplinglocs import UtilsSamplingLocs
from utilsio import UtilsIO
from utilsplotting import UtilsPlotting
from utilsopdims import UtilsOpDims
from matplotlib import pyplot as plt
from make_unit_length import make_unit_length
from get_neg_deltaFF import get_neg_deltaFF
from sklearn.decomposition import PCA
from remove_dimension_from_weight_matrix import remove_dimension_from_weight_matrix
from get_state_distance_between_trajs import get_state_distance_between_trajs
from get_mse import get_mse
import torch
import argparse
import sys
import os
import numpy as np
sys.path.append('../../')
from src.runner import run
from src.utils import load_config, set_seed
from src.constants import CONFIG_DIR, PROJECT_ROOT
from src.optimizer import optimizer_creator
from src.network.rnn import cRNN
from src.training.training_rnn_ext1 import RNNTrainingModule1
from datetime import datetime
#
from get_op_dim import get_weights
from get_op_dim import retrieve_op_dimensions
from get_op_dim import setup_environment
from scipy.linalg import subspace_angles
from matplotlib.ticker import MultipleLocator, FixedLocator

def get_global_operative_dimensions(inputfilename=None, ctxt="all"):
    network_type = 'ctxt'
    dim_type = 'columns'
    usl = UtilsSamplingLocs()
    uod = UtilsOpDims()
    n_units = 100

    # load  local operative dimensions
    sampling_loc_props = usl.get_sampling_location_properties(network_type='ctxt', skip_l=400)
    if inputfilename:
        pass
    else:     
        inputfilename = os.path.join(os.getcwd(), 'local_operative_dimensions', 'localOpDims_'+network_type+'_'+dim_type+'.h5')
    
    [all_local_op_dims, all_fvals] = uod.load_local_op_dims(inputfilename, n_units, sampling_loc_props, network_type='ctxt')

    # combine local operative dimensions to obtain global operative dimensions 
    sampling_locs_to_combine = ctxt #'all' # options for ctxt network: 'ctxt1' 'ctxt2' 'allPosChoice' 'allNegChoice'
    [global_op_dims, singular_values_of_global_op_dims] = uod.get_global_operative_dimensions(sampling_locs_to_combine, sampling_loc_props, all_local_op_dims, all_fvals)

    return global_op_dims, singular_values_of_global_op_dims

def plot_subspace_angle(ctxt1,ctxt2,n_global_dim, label_x, label_y, plot_name):

    global_op_dims_one, singular_values_of_global_op_dims_one = get_global_operative_dimensions(ctxt=ctxt1)
    global_op_dims_two, singular_values_of_global_op_dims_two = get_global_operative_dimensions(ctxt=ctxt2)
    
    angles = np.empty(global_op_dims_one.shape)
    rows, cols = global_op_dims_one.shape[0], global_op_dims_one.shape[1]
    for i in range(rows):
        for j in range(cols):
            angles[i,j] = np.rad2deg(subspace_angles(global_op_dims_one[:,i].reshape(-1,1), global_op_dims_two[:,j].reshape(-1,1)))[0]
    
    fig, ax = plt.subplots()
    plt.imshow(angles[:n_global_dim,:n_global_dim], cmap='viridis', interpolation='nearest', vmin=0, vmax=90, extent=[1,n_global_dim,n_global_dim,1])
    plt.xlabel(r'q$_{i}$ ('+ label_x +r'$_{1}$)')
    plt.ylabel(r'q$_{i}$ ('+ label_y +r'$_{2}$)')
    # Add colorbar
    cbar = plt.colorbar(label="angle (deg)", shrink=0.5)
    cbar.set_ticks(ticks=[0, 30, 60, 90])
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    plt.gca().invert_yaxis()
    fig.savefig(plot_name, bbox_inches="tight")

def get_ctxt_ids(the_model):
    ctxt1IDs, ctxt2IDs = np.empty(0), np.empty(0)
    for loader in [the_model.train_dataloader, the_model.test_dataloader, the_model.val_dataloader]:
        for i, (inputs, _) in enumerate(loader):
            contexts = inputs[:,-1,2].detach().numpy()
            ctxt1IDs = np.append(ctxt1IDs, contexts)
            contexts = inputs[:,-1,3].detach().numpy()
            ctxt2IDs = np.append(ctxt2IDs, contexts)

    # every trial is assigned
    assert( (ctxt2IDs == ((ctxt1IDs + 1) % 2)).all() ) 

    ctxt1IDs = np.where(ctxt1IDs == 1)
    ctxt2IDs = np.where(ctxt2IDs == 1)  

    return ctxt1IDs, ctxt2IDs

def get_choice_ids(the_model, w_in, w_hidden, w_out):
    forwardPass, targets = the_model.run_one_forwardPass_all_sets(w_in, w_hidden, w_out, noise_sigma=0)
    directions = forwardPass["m_z_t"][0,-10:,:].mean(axis=0) > 0
    choice1IDs = np.where(directions)
    choice2IDs = np.where(np.invert(directions))
    return choice1IDs, choice2IDs

def calculate_specific_mses(inputfilename, w_in, w_hidden, w_out, tm, op_dim_ctxt, first_IDs, second_IDs):
    # get global operative dim for ctxt 1
    global_op_dims, singular_values_of_global_op_dims = get_global_operative_dimensions(inputfilename, ctxt=op_dim_ctxt) 
    n_op_dims         = 100
    n_units           = 100
    mses_1     = np.full([n_op_dims, 1], np.nan)
    mses_2     = np.full([n_op_dims, 1], np.nan)

    for dim_nr in range(n_op_dims):
        # modify W
        n_Wrr_n_modified = remove_dimension_from_weight_matrix(w_hidden.detach().numpy(), global_op_dims[:,dim_nr+1:n_units+1], dim_type)

        # run modified network
        forwardPass, targets = tm.run_one_forwardPass_all_sets(w_in, n_Wrr_n_modified, w_out, noise_sigma=0)
        
        # get performance measures
        mses_1[dim_nr, 0] = get_mse(forwardPass["m_z_t"], targets, first_IDs)
        mses_2[dim_nr, 0] = get_mse(forwardPass["m_z_t"], targets, second_IDs)
    
    return mses_1, mses_2

def cost_plots_different_settings(inputfilename, weightpath, tm):
    uplt = UtilsPlotting()
    w_in, w_hidden, w_out = get_weights(path=weightpath)

    ctxt1_ids, ctxt2_ids = get_ctxt_ids(tm)
    choice1IDs, choice2IDs = get_choice_ids(tm, w_in, w_hidden, w_out)

    op_dim = "ctxt1"
    mses_ctxt1, mses_ctxt2 = calculate_specific_mses(inputfilename, w_in, w_hidden, w_out, tm, op_dim, ctxt1_ids, ctxt2_ids)

    op_dim = "allPosChoice"
    mses_choice1, mses_choice2 = calculate_specific_mses(inputfilename, w_in, w_hidden, w_out, tm, op_dim, choice1IDs, choice2IDs)

    # plot performance over reduced-rank Ws
    [fig, ax] = uplt.plot_lineplot(np.arange(100), mses_ctxt1, "network output cost for reduced-rank W", "rank(W$^{OP}_k$)", "cost")
    fig.savefig("ZZZ1.png", bbox_inches="tight")
    [fig, ax] = uplt.plot_lineplot(np.arange(100), mses_ctxt2, "network output cost for reduced-rank W", "rank(W$^{OP}_k$)", "cost")
    fig.savefig("ZZZ2.png", bbox_inches="tight")
    [fig, ax] = uplt.plot_lineplot(np.arange(100), mses_choice1, "network output cost for reduced-rank W", "rank(W$^{OP}_k$)", "cost")
    fig.savefig("ZZZ3.png", bbox_inches="tight")
    [fig, ax] = uplt.plot_lineplot(np.arange(100), mses_choice2, "network output cost for reduced-rank W", "rank(W$^{OP}_k$)", "cost")
    fig.savefig("ZZZ4.png", bbox_inches="tight")

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--config", help="Config file to use", default="rnn.yaml"
    )
    args = parser.parse_args()

    config = load_config(CONFIG_DIR / args.config)
    set_seed(config["experiment"]["seed"])

    """
    ctxt1 = "allPosChoice"
    ctxt2 = "allNegChoice"
    n_global_dim = 30
    label_x = "choice"
    label_y = "choice"
    plot_name = "choice.png"
    plot_subspace_angle(ctxt1,ctxt2,n_global_dim,label_x,label_y,plot_name)

    ctxt1 = "ctxt1"
    ctxt2 = "ctxt2"
    n_global_dim = 30
    label_x = "ctxt"
    label_y = "ctxt"
    plot_name = "ctxt_s.png"
    plot_subspace_angle(ctxt1,ctxt2,n_global_dim,label_x,label_y,plot_name)
    """
    network_type = 'ctxt'
    dim_type = 'columns'
    sampling_locs_to_combine = "all"
    inputfilename = os.path.join(os.getcwd(), 'local_operative_dimensions', 'localOpDims_'+network_type+'_'+dim_type+'.h5')
    path = r"../../io/output/rnn1/three_correct_model.pt"
 
    model, optimizer = None, None
    tm, optimizer = setup_environment(config["experiment"], path, model, optimizer)

    cost_plots_different_settings(inputfilename, path, tm)