"""Utility classes and functions.

Functions
---------
prepare_device(device="gpu")
save_torch_model(model, filename)
load_torch_model(model, filename)
get_config(exp_name)
get_model_name(expname, seed)
cubicFunc(x, intercept, slope_1, slope_2, slope_3)

Classes
---------
MetricTracker()

"""

import json
import yaml
import torch
import pandas as pd
import numpy as np
import xarray as xr
from utils.DIRECTORIES import DATA_DIRECTORY, MODEL_DIRECTORY

import scipy.stats as stats

def prepare_device(device="gpu"):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    if device == "gpu":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            print("Warning: MPS device not found." "Training will be performed on CPU.")
            device = torch.device("cpu")
    elif device == "cpu":
        device = torch.device("cpu")
    else:
        raise NotImplementedError

    return device


def save_torch_model(model, filename):
    if filename[-3:] != ".pt":
        filename = filename + ".pt"

    torch.save(model.state_dict(), MODEL_DIRECTORY + filename)


def load_torch_model(model, filename):
    model.load_state_dict(torch.load(MODEL_DIRECTORY + filename))
    model.eval()
    return model

def get_config(exp_name, filetype="yaml"):
    """
    Get the configuration for the experiment.
    
    Parameters
    ----------
    exp_name : str
        The name of the experiment.
    filetype : str, optional
        The type of configuration file to read ('json' or 'yaml'), by default 'json'.
    
    Returns
    -------
    dict
        The configuration dictionary.
    """
    
    if filetype == "json":
        return get_config_json(exp_name)
    elif filetype == "yaml":
        return get_config_yaml(exp_name)
    else:
        raise ValueError("Unsupported file type. Use 'json' or 'yaml'.")

def get_config_json(exp_name):

    basename = "exp"

    with open("config/config_" + exp_name[len(basename) :] + ".json") as f:
        config = json.load(f)

    assert config["expname"] == basename + exp_name[len(basename) :], "Exp_Name must be equal to config[exp_name]"

    assert len(config["datamaker"]["models"]) == len(config["datamaker"]["data_periods"])

    # add additional attributes for easier use later
    config["datamaker"]["fig_dpi"] = config["fig_dpi"]

    return config

def get_config_yaml(exp_name):

    try:
        with open("config/config_" + exp_name + ".yaml") as f:
            config = yaml.safe_load(f)

        assert len(config["datamaker"]["models"]) == len(config["datamaker"]["data_periods"]), "Length of models and data_periods must be equal."
    except FileNotFoundError:
        with open("config/data_" + exp_name + ".yaml") as f:
            config = yaml.safe_load(f)

    assert config["expname"] == exp_name, "Exp_Name must be equal to config[exp_name]"

    # add additional attributes for easier use later
    config["datamaker"]["fig_dpi"] = config["fig_dpi"]

    if not 'split_by_years' in config["datamaker"]:
        config["datamaker"]["split_by_years"] = False

    return config

def load_data(var, dp, config):
    if isinstance(dp, list):
        da_list = []
        for dp_ in dp:
            da_list.append(
                xr.open_dataarray(DATA_DIRECTORY + config["datafolder"] + var + dp_ +".nc")
            )
        da = xr.concat(da_list, dim='time')
    else:
        da = xr.open_dataarray(DATA_DIRECTORY + config["datafolder"] + var + dp +".nc")

    if config['detrend']:
        return da - da.mean(dim='member')
    else:
        return da


def get_model_name(exp_name, seed, suffix=""):
    return (exp_name + suffix + '_s' + str(seed))

def linear_regression(x, y):
    if y.ndim == 1:
        x_ = x - np.mean(x)
        x_var = np.mean(x_*x_)
        y_ = y - np.mean(y)

        a = np.mean(x_*y_)/x_var
        b = np.mean(y)-a*np.mean(x)

    elif y.ndim == 3:
        x_ = x[:,np.newaxis,np.newaxis] - np.mean(x)
        x_var = np.mean(x_*x_)
        y_ = y - np.mean(y,axis=0,keepdims=True)

        a = np.mean(x_*y_,axis=0)/x_var
        b = np.mean(y,axis=0)-a*np.mean(x)

    elif y.ndim == 4:
        x_ = x[:,np.newaxis,np.newaxis,np.newaxis] - np.mean(x)
        x_var = np.mean(x_*x_)
        y_ = y - np.mean(y,axis=0,keepdims=True)

        a = np.mean(x_*y_,axis=0)/x_var
        b = np.mean(y,axis=0)-a*np.mean(x)

    else:
        raise RuntimeError("Check dimensions.")
    
    return a, b

def linear_regression_with_confidence_bounds(x, y, confidence=0.1):
    """
    Compute linear regression of x and y with confidence bounds.

    Parameters:
        x (array-like): Independent variable data.
        y (array-like): Dependent variable data.
        confidence (float, optional): Confidence level for the bounds. Default is 0.95.

    Returns:
        slope (float): Slope of the regression line.
        intercept (float): Intercept of the regression line.
        slope_stderr (float): Standard error of the slope.
        intercept_stderr (float): Standard error of the intercept.
        confidence_bounds (tuple): Lower and upper confidence bounds of the regression line.
    """
    # Compute linear regression
    res = stats.linregress(x, y)
    slope = res.slope
    intercept = res.intercept
    
    # Degrees of freedom
    df = len(x) - 2
    
    # T-statistic for given confidence level
    t_value = stats.t.ppf(confidence / 2, df)
        
    # Standard error of the slope and intercept
    slope_stderr = res.stderr
    intercept_stderr = res.intercept_stderr
    
    return slope, intercept, t_value * slope_stderr, t_value * intercept_stderr

def write_line(f, x, y):
    f.write(str(x))
    f.write(" ")
    f.write(str(y))
    f.write("\n")
    
def write_lines(fname,x,y):
    with open(fname,"w") as f:
        for xx,yy in zip(x,y):
            write_line(f, xx, yy)

class MetricTracker:
    """Could have written this as a subclass of dict() itself, but instead it can now
    hold other attributes if desired.
    """

    def __init__(self, *keys):

        self.history = dict()
        for k in keys:
            self.history[k] = []
        self.reset()

    def reset(self):
        for key in self.history:
            self.history[key] = []

    def update(self, key, value):
        if key in self.history:
            self.history[key].append(value)

    def result(self):
        for key in self.history:
            self.history[key] = np.nanmean(self.history[key])

    def print(self, idx=None):
        for key in self.history.keys():
            if idx is None:
                print(f"  {key} = {self.history[key]:.5f}")
            else:
                print(f"  {key} = {self.history[key][idx]:.5f}")
