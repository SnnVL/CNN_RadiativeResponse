from torch.utils.data import Dataset
import torch
import numpy as np


class MapToValueData(Dataset):
    """
    Custom dataset for torch data.
    """

    def __init__(self, dict_data):

        self.input = dict_data["x"]
        self.target = dict_data["y"]

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, idx):

        input = self.input[idx, ...]
        target = self.target[idx]

        return (
            torch.tensor(input, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )
    
class MapToMapData(Dataset):
    """
    Custom dataset for torch data.
    """

    def __init__(self, dict_data):

        self.input = dict_data["x"]
        self.target = dict_data["y"]

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, idx):

        input = self.input[idx, ...]
        target = self.target[idx, ...]

        return (
            torch.tensor(input, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )