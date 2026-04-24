import torch.nn.functional as F
import torch
import numpy as np


class mse_sphere(torch.nn.Module):
    """
    Negative log likelihood loss for a SHASH distribution.
    """

    def __init__(self, lat, lon, device, mask=None):
        super(mse_sphere, self).__init__()

        weights = np.array([np.cos(np.deg2rad(lat))]).T
        weights = np.tile(weights, (1, lon.size))
        weights = weights/np.sum(weights)
        weights = weights.astype(np.float32)

        if mask is not None:
            weights = weights * mask

        weights = torch.from_numpy(weights)

        self.weights = weights.to(device)

    def forward(self, output, target):
        
        # Compute MSE over lat/lon
        mse = torch.sum(self.weights * (output - target) ** 2, dim = (-1,-2))

        return mse.mean()


def mse_loss(output, target):
    return F.mse_loss(output, target)

def mae_loss(output, target):
    return F.l1_loss(output, target)
