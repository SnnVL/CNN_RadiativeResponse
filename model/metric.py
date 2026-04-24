import torch
import numpy as np

import xarray as xr
with xr.open_dataarray("/Users/senne/Documents/data_PT/shapefiles/mask_all.nc") as f:
    lat = f.lat.values
    lon = f.lon.values

weights = np.array([np.cos(np.deg2rad(lat))]).T
weights = np.tile(weights, (1, lon.size))
weights = weights/np.sum(weights)
weights = weights.astype(np.float32)
weights = torch.from_numpy(weights)


def custom_mae(output, target):
    """Compute the prediction mean absolute error.
    The "predicted value" is the median of the conditional distribution.

    """

    assert len(output[:, 0]) == len(target)

    return np.mean(np.abs(output - target))

def r2_score(output, target):
    """Compute the R^2 score of the prediction.
    The "predicted value" is the median of the conditional distribution.

    """

    assert len(output[:, 0]) == len(target)

    # Compute R^2 score
    with torch.no_grad():
        x = target.to('cpu')
        y = output.to('cpu')
        ss_res = torch.sum((x - y) ** 2)
        ss_tot = torch.sum((x - torch.mean(x)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

    return r2

def pattern_correlation(output, target):
    """Compute the pattern correlation of the prediction.
    The "predicted value" is the median of the conditional distribution.

    """

    with torch.no_grad():
        x = target.to('cpu')
        y = output.to('cpu')

        # Compute R^2 score
        x_avg = torch.sum(weights * x, dim = (-1,-2), keepdim=True)
        y_avg = torch.sum(weights * y, dim = (-1,-2), keepdim=True)

        x_anom = x - x_avg
        y_anom = y - y_avg

        x_std = torch.sqrt(torch.sum(weights * x_anom**2, dim = (-1,-2)))
        y_std = torch.sqrt(torch.sum(weights * y_anom**2, dim = (-1,-2)))

        s_xy = torch.sum(weights * x_anom*y_anom, dim = (-1,-2))

        r = torch.mean(s_xy/(x_std * y_std))

    return r


def mse_sphere(output, target):

    with torch.no_grad():
        x = target.to('cpu')
        y = output.to('cpu')

        # Compute MSE over lat/lon
        mse = torch.sum(weights * (x - y) ** 2, dim = (-1,-2))

        # Compute MSE over batch
        mse = torch.mean(mse)
        
    return mse


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
