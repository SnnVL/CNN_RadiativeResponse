"""
Loading data and models
"""

import torch
import numpy as np
import random
import xarray as xr

from utils import utils
from data_loader.data_generator import ClimateData, ObsData
from data_loader.data_loaders import MapToMapData, MapToValueData
from model.model import ConvNet, LinearNet, GlobalAverageNet
from utils.DIRECTORIES import SHAPE_DIRECTORY
import shap

def load_model_and_data(conf_name, seed, verbose=True, config=None):
    if config is None:
        config = utils.get_config(conf_name)

    # make model initialization deterministic
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    model_name = utils.get_model_name(config["expname"], seed)
    if verbose:
        print("___________________")
        print(model_name)

    if "subtract_val" in config['datamaker'].keys():
        if config['datamaker']['subtract_val'] == 'save':
            config['datamaker']['subtract_val'] = model_name + ".pickle"

    # Get the Data
    if verbose:
        print("___________________")
        print("Get the data.")
    data = ClimateData(
        config["datamaker"],
        expname=config["expname"],
        seed=seed,
        verbose=verbose,
    )

    if config["datamaker"]["map_output"]:
        trainset = MapToMapData(data.d_train)
        valset = MapToMapData(data.d_val)
        testset = MapToMapData(data.d_test)
    else:
        trainset = MapToValueData(data.d_train)
        valset = MapToValueData(data.d_val)
        testset = MapToValueData(data.d_test)

    # Setup the Model
    if verbose:
        print("___________________")
        print("Loading the model.")

    if config["arch"]["type"] == 'LinearNet':
        model = LinearNet(
            config=config["arch"],
        )
    elif config["arch"]["type"] == 'ConvNet':
        model = ConvNet(
            config=config["arch"],
        )
    else:
        raise NotImplementedError("Model type not implemented.")
    model = utils.load_torch_model(model, model_name + ".pt")

    return config, model, data, trainset, valset, testset


def load_data(
        conf_name, 
        seed, 
        verbose=True, 
        subtract_val = None, 
        input_mask = None, 
        anomaly_dates = None, 
        config = None
    ):

    if config is None:
        config = utils.get_config(conf_name)

    # make model initialization deterministic
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    if "subtract_val" in config['datamaker'].keys():
        if config['datamaker']['subtract_val'] == 'save':
            config['datamaker']['subtract_val'] = 'load'
    if subtract_val is not None:
        config['datamaker']['subtract_val'] = subtract_val
    if anomaly_dates is not None:
        config['datamaker']['anomaly_dates'] = anomaly_dates
    if input_mask is not None:
        config['datamaker']['input_mask'] = input_mask

    # Get the Data
    if verbose:
        print("___________________")
        print("Get the data.")
    data = ClimateData(
        config["datamaker"],
        expname=config["expname"],
        seed=seed,
        verbose=verbose,
    )

    if config["datamaker"]["map_output"]:
        trainset = MapToMapData(data.d_train)
        valset = MapToMapData(data.d_val)
        testset = MapToMapData(data.d_test)
    else:
        trainset = MapToValueData(data.d_train)
        valset = MapToValueData(data.d_val)
        testset = MapToValueData(data.d_test)

    return config, data, trainset, valset, testset

def load_obs_data(
        conf_name,
        seed,
        verbose=True,
        subtract_val = None,
        anomaly_dates = None,
        input_mask = None,
        config = None
    ):

    if config is None:
        config = utils.get_config(conf_name)

    # make model initialization deterministic
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    if subtract_val is not None:
        config['datamaker']['subtract_val'] = subtract_val
    if anomaly_dates is not None:
        config['datamaker']['anomaly_dates'] = anomaly_dates
    if input_mask is not None:
        config['datamaker']['input_mask'] = input_mask

    # Get the Data
    if verbose:
        print("___________________")
        print("Get the data.")
    data = ObsData(
        config["datamaker"],
        expname=config["expname"],
        seed=seed,
        verbose=verbose,
    )

    dataset = MapToValueData(data.d_obs)

    return config, data, dataset

def get_gradient(model, dataset=None, dataloader=None, batch_size=128, device="cpu"):

    if (dataset is None) & (dataloader is None):
        raise ValueError("both dataset and dataloader cannot be done.")

    if (dataset is not None) & (dataloader is not None):
        raise ValueError("dataset and dataloader cannot both be defined. choose one.")

    if dataset is not None:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

    model.to(device)
    model.eval()
    grads = None
    for batch_idx, (input, target) in enumerate(dataloader):
        input, target = (
            input.to(device),
            target.to(device),
        )

        model.zero_grad()
        input.requires_grad_(True)
        out = model(input)
        out.backward(torch.ones_like(out))

        if grads is None:
            grads = input.grad.to('cpu').numpy()
        else:
            grads = np.concatenate((grads, input.grad.to('cpu').numpy()), axis=0)

    return grads



def get_global_avg_model(model, weights=None, input_size=None, bias=False):

    if weights is None and input_size is None:
        raise ValueError("Either weights or input_size must be provided.")
    if input_size is None :
        input_size = weights.shape[0]*weights.shape[1]
    if weights is None:
        weights = np.ones((1, input_size))
    
    weights = weights / np.sum(weights)
    avg_weights = torch.from_numpy(weights.flatten().reshape((1,input_size)))

    config_avg = {
        "input_size": input_size,
        "bias": bias,
    }

    # Create linear model layer
    avg_model = GlobalAverageNet(
        model,
        config=config_avg,
    )
    state_dict = avg_model.state_dict()

    # Access the layer's state dictionary
    state_dict = avg_model.state_dict()

    # Replace the weights in the state dictionary with the new tensor
    state_dict['out.weight'] = avg_weights

    # Load the modified state dictionary back into the layer
    avg_model.load_state_dict(state_dict)

    return avg_model

def make_predictions(model, dataset=None, dataloader=None, batch_size=128, device="cpu"):

    if (dataset is None) & (dataloader is None):
        raise ValueError("both dataset and dataloader cannot be done.")

    if (dataset is not None) & (dataloader is not None):
        raise ValueError("dataset and dataloader cannot both be defined. choose one.")

    if dataset is not None:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

    model.to(device)
    model.eval()
    with torch.inference_mode():
        output = None
        for batch_idx, (input, target) in enumerate(dataloader):
            input, target = (
                input.to(device),
                target.to(device),
            )

            out = model(input).to("cpu").numpy()
            if output is None:
                output = out
            else:
                output = np.concatenate((output, out), axis=0)

    return output


def get_global_mean_input(config, seed, weights=None):

    config_nomask = config.copy()
    config_nomask["datamaker"]["input_mask"] = False
    data = ClimateData(
        config["datamaker"],
        expname=config["expname"]+"_nomask",
        seed=seed,
        verbose=False,
    )

    if weights is None:
        weights = np.array([np.cos(np.deg2rad(data.input_lat))]).T
        weights = np.tile(weights, (1, data.input_lon.shape[0]))
        if config["datamaker"]["input_mask"]:
            input_mask = xr.open_dataarray(SHAPE_DIRECTORY + config["datamaker"]["input_mask"]).values
            weights = weights * input_mask
        weights = weights/np.sum(weights)

    gm_train = np.sum(data.d_train['x']*weights[None,None,:,:],axis=(-1,-2))
    gm_train_val = np.sum(data.d_val['x']*weights[None,None,:,:],axis=(-1,-2))
    gm_test = np.sum(data.d_test['x']*weights[None,None,:,:],axis=(-1,-2))

    return gm_train, gm_train_val, gm_test, weights

def get_global_mean_obs(config, seed, weights=None):

    config_nomask = config.copy()
    config_nomask["datamaker"]["input_mask"] = False
    data = ObsData(
        config["datamaker"],
        expname=config["expname"]+"_nomask",
        seed=seed,
        verbose=False,
    )

    if weights is None:
        weights = np.array([np.cos(np.deg2rad(data.input_lat))]).T
        weights = np.tile(weights, (1, data.input_lon.shape[0]))
        if config["datamaker"]["input_mask"]:
            input_mask = xr.open_dataarray(SHAPE_DIRECTORY + config["datamaker"]["input_mask"]).values
            weights = weights * input_mask
        weights = weights/np.sum(weights)

    gm_obs = np.sum(data.d_obs['x']*weights[None,None,:,:],axis=(-1,-2))

    return gm_obs, weights


def deep_shap(model, dataset = None, dataloader = None, baseline=None, batch_size=128, device="cpu"):

    if (dataset is None) & (dataloader is None):
        raise ValueError("both dataset and dataloader cannot be done.")

    if (dataset is not None) & (dataloader is not None):
        raise ValueError("dataset and dataloader cannot both be defined. choose one.")

    # Create dataloader from dataset
    if dataset is not None:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

    # Create baseline if not provided
    if baseline is None:
        input_size = next(iter(dataloader))[0].shape[1:]
        baseline = np.zeros(input_size).astype(np.float32)
        baseline = baseline[np.newaxis,...]

    # Initialize DeepExplainer
    model.to(device)
    model.eval()
    de = shap.DeepExplainer(model, torch.tensor(baseline, dtype=torch.float32, device=device))

    shap_vals = None
    for batch_idx, (input, target) in enumerate(dataloader):
        input, target = (
            input.to(device),
            target.to(device),
        )

        out = de.shap_values(input, check_additivity=True)
        if shap_vals is None:
            shap_vals = out
        else:
            shap_vals = np.concatenate((shap_vals, out), axis=0)

    return np.squeeze(shap_vals)