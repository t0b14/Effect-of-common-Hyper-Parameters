from samplinglocs import UtilsSamplingLocs
from utilsio import UtilsIO
from utilsplotting import UtilsPlotting
from utilsopdims import UtilsOpDims
from matplotlib import pyplot as plt
import torch
import argparse
import sys
import os
import numpy as np
sys.path.append('../../')
from src.runner import run
from src.utils import load_config, set_seed
from src.constants import CONFIG_DIR, DATA_DIR
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

def calc_operative_dimensions(sampling_locs, sampling_loc_props, n_units, USL, UOD, UIO):

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputfilename = os.path.join(DATA_DIR, 'local_operative_dimensions', 'localOpDims_'+time_stamp+'.h5')

    n_dims_to_find  = 100
    n_inputs=4
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
            
        # collect trajs for full-rank network as reference
        samplingLocParams["all_trajs_org"] = np.full([n_units, 2], np.nan)
        inputs_relax = np.zeros([n_inputs, 1, 1])
        ctxt_id = int(samplingLocParams["ctxt_id"])
        inputs_relax[1+ctxt_id, :, :] = 0
        conditionIds_relax = np.reshape(np.asarray([ctxt_id, ctxt_id]), [1,2])
        # init_n_x0_c = np.reshape(np.asarray([samplingLocParams.sampling_loc; samplingLocParams.sampling_loc]), [n_units, 2])
        init_n_x0_c = np.concatenate([samplingLocParams["sampling_loc"], samplingLocParams["sampling_loc"]], axis=1)
        
        net_noise_trajs = 0
        forwardPass_modified = run_one_forwardPass(n_Wru_v, n_Wrr_n, m_Wzr_n, init_n_x0_c, n_bx_1, m_bz_1, inputs_relax, conditionIds_relax, seed_run, net_noise_trajs)
        
        # add first step separately and then all the other steps
        samplingLocParams["all_trajs_org"][:, 0] = np.reshape(forwardPass_modified["n_x0_1"][:, 0], [n_units, ])
        samplingLocParams["all_trajs_org"][:, 1] = np.reshape(forwardPass_modified["n_x_t"], [n_units, ])

        # always columns type
        samplingLocParams["local_op_dims"][0, :] = make_unit_length(np.matmul(n_Wrr_n, np.tanh(samplingLocParams["sampling_loc"]))).T

        dims_to_be_orth = np.zeros([100,0])
        fval = get_neg_deltaFF(samplingLocParams["local_op_dims"][0, :].T, dims_to_be_orth, samplingLocParams, n_Wru_v, n_Wrr_n, m_Wzr_n, n_bx_1, m_bz_1, dim_type, network_type)
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
def retrieve_op_dimensions(config, weights=None, path=None, model=None, optimizer=None):
    
    w_in, w_hidden, w_out = get_weights(weights, path)

    tm, optimizer = setup_environment(config, path, model, optimizer)

    # run 
    activity, coherencies, conditionIds = tm.get_activity_and_data_for_op_dimension()

    activity = torch.permute(activity, (2, 1, 0)).detach().numpy()

    usl = UtilsSamplingLocs()
    sampling_locs_props = usl.get_sampling_location_properties(skip_l = 400)

    uio = UtilsIO()
    uplt = UtilsPlotting()
    uod = UtilsOpDims()

    sampling_locs = usl.get_sampling_locs_on_condAvgTrajs_ctxt(activity, sampling_locs_props, conditionIds, coherencies)
    [fig, ax] = uplt.plot_sampling_locs_on_condAvgTrajs(activity, sampling_locs, sampling_locs_props, [], conditionIds, coherencies, network_type="ctxt") 
    
    fig.savefig('op_dim_PC.png')
    
    calc_operative_dimensions(sampling_locs, sampling_locs_props, 100, usl, uod, uio)


    print("--- end of retrieve op")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--config", help="Config file to use", default="rnn.yaml"
    )
    args = parser.parse_args()

    config = load_config(CONFIG_DIR / args.config)
    set_seed(config["experiment"]["seed"])

    path = r"..\..\io\output\rnn1\one_model.pt"
    retrieve_op_dimensions(config["experiment"], path=path)