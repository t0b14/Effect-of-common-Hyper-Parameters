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

def get_weights(weights=None, path=None):
    assert( not ((weights == None) and (path == None)) )
    
    if path:    
        model = torch.load(path)
        for name, values in model.items():
            if name == "rnn.W_in":
                w_in = values
            elif name == "rnn.W_hidden":
                w_hidden = values
            elif name == "fc_out.weight":
                w_out = values
    if weights:
        [w_in, w_hidden, w_out] = weights 
    return w_in, w_hidden, w_out

def calc_operative_dimensions(tm, sampling_locs, sampling_loc_props, n_Wru_v, n_Wrr_n, m_Wzr_n, n_units, USL, UOD, UIO):
    
    network_type = 'ctxt'
    dim_type = 'columns'
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputfilename = os.path.join(os.getcwd(), 'local_operative_dimensions', 'localOpDims_'+network_type+'_'+dim_type+'_'+time_stamp+'.h5')

    n_dims_to_find = 100
    n_inputs = 4
    n_sampling_locs = np.size(sampling_loc_props["t_start_pt_per_loc"]);
    for loc_nr in range(n_sampling_locs):
        samplingLocParams = {}
        samplingLocParams["local_op_dims"] = np.full([n_dims_to_find, n_units], np.nan)
        samplingLocParams["all_fvals"]     = np.full([n_dims_to_find, 1], np.nan)
        # get sampling location
        start_pt_nr = USL.map_optLocNr_to_startPtNr(loc_nr, sampling_loc_props)
        inpCond_nr  = USL.map_optLocNr_to_inpCondNr(loc_nr, sampling_loc_props, network_type="ctxt")    
        samplingLocParams["sampling_loc"]  = np.reshape(sampling_locs[inpCond_nr, :, start_pt_nr], [n_units, 1])
        
        # info on sampling location
        samplingLocParams["t_start_point"] = sampling_loc_props["t_start_pt_per_loc"][0,loc_nr]
        samplingLocParams["ctxt_id"]  = sampling_loc_props["ctxt_per_loc"][0,loc_nr]
        samplingLocParams["signCoh1"] = sampling_loc_props["signCoh1_per_loc"][0,loc_nr]
        samplingLocParams["signCoh2"] = sampling_loc_props["signCoh2_per_loc"][0,loc_nr]

        dim_name = UOD.get_name_of_local_operative_dims(network_type="ctxt", t_start_point=samplingLocParams["t_start_point"], ctxt_id=samplingLocParams["ctxt_id"], 
                                                        signCoh1=samplingLocParams["signCoh1"], signCoh2=samplingLocParams["signCoh2"])
        samplingLocParams["all_trajs_org"] = np.full([n_units, 2], np.nan)
        inputs_relax = np.zeros([n_inputs, 1, 1])

        init_n_x0_c = torch.tensor(samplingLocParams["sampling_loc"], dtype=torch.float32).reshape(1,-1)

        net_noise_trajs = 0
        forwardPass_modified = tm.run_one_forwardPass(n_Wru_v, n_Wrr_n, m_Wzr_n, init_n_x0_c, net_noise_trajs, torch.tensor(inputs_relax, dtype=torch.float32).reshape(1, 1, -1))
        
        # add first step separately and then all the other steps    
        samplingLocParams["all_trajs_org"][:, 0] = np.reshape(forwardPass_modified["n_x0_1"], [n_units, ])
        samplingLocParams["all_trajs_org"][:, 1] = np.reshape(forwardPass_modified["n_x_t"], [n_units, ])

        # always columns type
        samplingLocParams["local_op_dims"][0, :] = make_unit_length(np.matmul(n_Wrr_n.T.detach().numpy() , np.tanh(samplingLocParams["sampling_loc"]))).T 


        dims_to_be_orth = np.zeros([100,0])
        fval = get_neg_deltaFF(tm, samplingLocParams["local_op_dims"][0, :].T, dims_to_be_orth, samplingLocParams, n_Wru_v, n_Wrr_n, m_Wzr_n, dim_type="columns", network_type= "ctxt")

        samplingLocParams["all_fvals"] = np.full([n_units, 1], np.nan)
        samplingLocParams["all_fvals"][0, 0] = fval
    
        # save to hdf5

        hdf5_group_name = dim_name
        
        UIO.save_to_hdf5(outputfilename, hdf5_group_name, samplingLocParams)
    
    print("Done! All local operative dimensions saved to: "+ outputfilename) 

def setup_environment(config, path=None, model=None, optimizer=None):
    params = config["model"]
    if not model:
        model = cRNN(
                config["model"],
                input_s=params["in_dim"],
                output_s=params["out_dim"],
                hidden_s=params["hidden_dims"],
                hidden_noise=params["hidden_noise"]
                )
        model.load_state_dict(torch.load(path))
        model.eval()
    if not optimizer:
        optimizer = optimizer_creator(model.parameters(), config["optimizer"])

    tm = RNNTrainingModule1(model, optimizer, config)

    return tm, optimizer
# calc op dimension with weights or path to model
# S2
def retrieve_op_dimensions(path, tm):
    
    w_in, w_hidden, w_out = get_weights(path=path)

    # run 
    activity, coherencies, conditionIds = tm.get_activity_and_data_for_op_dimension([w_in, w_hidden, w_out])
    activity = torch.permute(activity, (2, 1, 0)).detach().numpy()

    usl = UtilsSamplingLocs()
    sampling_locs_props = usl.get_sampling_location_properties(skip_l = 400)

    uio = UtilsIO()
    uplt = UtilsPlotting()
    uod = UtilsOpDims()

    sampling_locs = usl.get_sampling_locs_on_condAvgTrajs_ctxt(activity, sampling_locs_props, conditionIds, coherencies)
    [fig, ax] = uplt.plot_sampling_locs_on_condAvgTrajs(activity, sampling_locs, sampling_locs_props, [], conditionIds, coherencies, network_type="ctxt") 
    
    fig.savefig('op_dim_PC.png')

    calc_operative_dimensions(tm, sampling_locs, sampling_locs_props, w_in, w_hidden, w_out, 100, usl, uod, uio)

    print("--- end of retrieve op")
# S1
def plot_dimensionality_high_variance_dim_W(path, tm):

    n_units = 100
    w_in, w_hidden, w_out = get_weights(path=path)
    UPlt = UtilsPlotting()
    # plot dimensionality of high-variance dimensions (perform SVD(W))
    _, S, _ = np.linalg.svd(w_hidden)
    S = np.square(S)
    [fig, ax] = UPlt.plot_lineplot(np.arange(n_units), S/np.sum(S)*100, "Dimensionality W", "PC(W)$_i$", "variance explained (%)")
    fig.savefig('dimensionality_of_high_variance_dimensions_of_W.png')
    # 

    
    activity, coherencies, conditionIds = tm.get_activity_and_data_for_op_dimension([w_in, w_hidden, w_out])
    activity = torch.permute(activity, (2, 1, 0)).detach().numpy()
    # plot dimensionality of network activities
    net_activities = np.reshape(activity, [n_units, -1])
    pca = PCA(n_components=n_units)
    pca.fit(net_activities.T)  # [n_samples, n_features]
    [fig, ax] = UPlt.plot_lineplot(np.arange(n_units), pca.explained_variance_ratio_, "Dimensionality X", "PC(X)$_i$", "variance explained (%)")
    fig.savefig('dim_network_activity.png')
    #

    activity_full_rank = activity
    # get high-variance dimensions
    [U, _, _] = np.linalg.svd(w_hidden.T.detach().numpy())

    # run network with reduced-rank W and collect performance measures
    # ( = mean squared error (mse) & State distance between full-rank and reduced-rank network trajectories)
    n_high_var_dims   = n_units
    mses              = np.full([n_high_var_dims, 1], np.nan)
    statedists_to_org = np.full([n_high_var_dims, 1], np.nan)
    for dim_nr in range(n_high_var_dims):
        
        # modify W
        n_Wrr_n_modified = remove_dimension_from_weight_matrix(w_hidden.T.detach().numpy(), U[:,dim_nr+1:n_units], 'columns')

        # run modified network
        forwardPass, targets = tm.run_one_forwardPass_on_val_set(w_in, n_Wrr_n_modified, w_out)

        # get performance measures
        mses[dim_nr, 0] = get_mse(forwardPass["m_z_t"], targets, 'all')
        statedists_to_org[dim_nr, 0] = get_state_distance_between_trajs(forwardPass["n_x_t"], activity_full_rank)

    # plot
    [fig, ax] = UPlt.plot_lineplot(np.arange(n_units), mses, "network output cost for reduced-rank W", "rank(W$^{PC}_k$)", "cost")
    fig.savefig('network_output_cost_r_r_W.png')
    [fig, ax] = UPlt.plot_lineplot(np.arange(n_units), statedists_to_org, "state distance to trajectory of full-rank W", "rank($W^{PC}_k$)", "state distance (a.u.)")
    fig.savefig('state_distance_to_traj_of_full_r_W.png')

#S3
def analyse_global_operative_dimensions(path, tm):
    network_type = 'ctxt'
    dim_type = 'columns'
    UPlt = UtilsPlotting()
    usl = UtilsSamplingLocs()
    uod = UtilsOpDims()
    n_units = 100

    # load  local operative dimensions
    sampling_loc_props = usl.get_sampling_location_properties(network_type='ctxt', skip_l=400)
    inputfilename = os.path.join(os.getcwd(), 'local_operative_dimensions', 'localOpDims_'+network_type+'_'+dim_type+'.h5')
    
    [all_local_op_dims, all_fvals] = uod.load_local_op_dims(inputfilename, n_units, sampling_loc_props, network_type='ctxt')

    # combine local operative dimensions to obtain global operative dimensions 
    sampling_locs_to_combine = 'all' # options for ctxt network: 'ctxt1' 'ctxt2' 'allPosChoice' 'allNegChoice'

    [global_op_dims, singular_values_of_global_op_dims] = uod.get_global_operative_dimensions(sampling_locs_to_combine, sampling_loc_props, all_local_op_dims, all_fvals)
    
    # plot dimensionality of global operative dimensions
    var_of_global_op_dims = np.square(singular_values_of_global_op_dims)
    [fig, ax] = UPlt.plot_lineplot(np.arange(n_units), (var_of_global_op_dims/np.sum(var_of_global_op_dims)*100), "Dimensionality of global operative dimensions", "PC(L)$_i$", "variance explained (#)")
    fig.savefig("dimensionality_of_global_operative_dimensions.png")

    ###  Performance over sequentially removing global operative dimensions
    w_in, w_hidden, w_out = get_weights(path=path)

    # run full-rank network as reference to calculate state distance measure
    forwardPass_org, targets = tm.run_one_forwardPass_on_val_set(w_in, w_hidden, w_out, 0)

    # run network with reduced-rank W and collect performance measures
    # ( = mean squared error (mse) & State distance between full-rank and reduced-rank network trajectories)
    n_op_dims         = n_units
    mses              = np.full([n_op_dims, 1], np.nan)
    statedists_to_org = np.full([n_op_dims, 1], np.nan)
    for dim_nr in range(n_op_dims):
        # modify W
        n_Wrr_n_modified = remove_dimension_from_weight_matrix(w_hidden.T, global_op_dims[:,dim_nr+1:n_units+1], dim_type)

        # run modified network
        forwardPass, targets = tm.run_one_forwardPass_on_val_set(w_in, n_Wrr_n_modified, w_out, 0)

        # get performance measures
        mses[dim_nr, 0] = get_mse(forwardPass["m_z_t"], targets, 'all')
        statedists_to_org[dim_nr, 0] = get_state_distance_between_trajs(forwardPass["n_x_t"], forwardPass_org["n_x_t"])

    # plot performance over reduced-rank Ws
    [fig, ax] = UPlt.plot_lineplot(np.arange(n_units), mses, "network output cost for reduced-rank W", "rank(W$^{OP}_k$)", "cost")
    fig.savefig("output_cost_for_reduced_rank_W.png")
    [fig, ax] = UPlt.plot_lineplot(np.arange(n_units), statedists_to_org, "state distance to trajectory of full-rank W", "rank(W$^{OP}_k$)", "state distance (a.u.)")
    fig.savefig("trajectory_of_full_rank_W.png")

#S4
def plot_various_g_op_dim(path, tm):
    network_type = 'ctxt'
    dim_type = 'columns'
    UPlt = UtilsPlotting()
    usl = UtilsSamplingLocs()
    uod = UtilsOpDims()
    n_units = 100

    # load  local operative dimensions
    sampling_loc_props = usl.get_sampling_location_properties(network_type='ctxt', skip_l=400)
    inputfilename = os.path.join(os.getcwd(), 'local_operative_dimensions', 'localOpDims_'+network_type+'_'+dim_type+'.h5')
    
    [all_local_op_dims, all_fvals] = uod.load_local_op_dims(inputfilename, n_units, sampling_loc_props, network_type='ctxt')

    # combine local operative dimensions to obtain global operative dimensions 
    sampling_locs_to_combine = 'all' # options for ctxt network: 'ctxt1' 'ctxt2' 'allPosChoice' 'allNegChoice'
    [global_op_dims, singular_values_of_global_op_dims] = uod.get_global_operative_dimensions(sampling_locs_to_combine, sampling_loc_props, all_local_op_dims, all_fvals)


     # & COMPARE NETWORK OUTPUT AND CONDITION AVERAGE TRAJECTORIES
    rankW = 15
    w_in, w_hidden, w_out = get_weights(path=path)
    # run full-rank network
    forwardPass_fullRank, targets = tm.run_one_forwardPass_on_val_set(w_in, w_hidden, w_out, 0)

    # run reduced-rank network
    n_Wrr_n_modified = remove_dimension_from_weight_matrix(w_hidden, global_op_dims[:, rankW:n_units+1], dim_type)
    forwardPass_reducedRank, targets = tm.run_one_forwardPass_on_val_set(w_in, n_Wrr_n_modified, w_out, 0)    

    # plot network outputs for several trials
    for trial_nr in np.arange(1,25,5):
        y_data = np.concatenate([np.reshape(targets[0,:,trial_nr], [-1, 1]), 
                                np.reshape(forwardPass_fullRank["m_z_t"][0, :, trial_nr], [-1, 1]), 
                                np.reshape(forwardPass_reducedRank["m_z_t"][0, :, trial_nr], [-1, 1])], axis=1)
        [fig, ax] = UPlt.plot_lineplot(range(np.shape(targets)[1]), y_data, "network output", "t", "z$_t$", display_names=["target", "full-rank", "reduced-rank"])
        ax.set_ylim(bottom=-1.1, top=1.1)
        fig.savefig("compare_network_output" + str(trial_nr) +".png")

    # plot one example trials for network trajectory
    activity, coherencies, conditionIds = tm.get_activity_and_data_for_op_dimension([w_in, w_hidden, w_out])

    [fig, ax] = UPlt.plot_full_and_reduced_rank_condAvgTrajs(forwardPass_fullRank["n_x_t"], forwardPass_reducedRank["n_x_t"], 
                    sampling_loc_props, [], conditionIds, coherencies, network_type, rankW)
    fig.savefig("compare_condition_average_trajectories.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--config", help="Config file to use", default="rnn.yaml"
    )
    args = parser.parse_args()

    config = load_config(CONFIG_DIR / args.config)
    set_seed(config["experiment"]["seed"])

    path = r"..\..\io\output\rnn1\one_model.pt"

    model, optimizer = None, None
    tm, optimizer = setup_environment(config["experiment"], path, model, optimizer)

    # S1 
    #plot_dimensionality_high_variance_dim_W(config["experiment"], path=path, tm)
    #print("finished S1")

    # S2
    #retrieve_op_dimensions(path, tm)
    #print("finished S2")

    # S3
    #analyse_global_operative_dimensions(path, tm)
    #print("finished S3")
    
    # S4
    plot_various_g_op_dim(path, tm)
    print("finished S4")
    






