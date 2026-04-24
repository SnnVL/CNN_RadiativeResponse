import numpy as np
import torch
import torch.nn as nn

__author__ = "Senne Van Loon"
__version__ = "06 May 2025"

# Acknowledgements:
# https://github.com/FrancescoSaverioZuppichini/Pytorch-how-and-when-to-use-Module-Sequential-ModuleList-and-ModuleDict
# https://github.com/eabarnes1010/pytorch_template


def conv_block(in_f, out_f, act_fun, pool_size, kernel_size, *args, **kwargs):    
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, kernel_size, *args, **kwargs),
        getattr(torch.nn, act_fun)(),
        torch.nn.MaxPool2d(kernel_size=(pool_size, pool_size), ceil_mode=True),
    )

def conv_sequence(\
        in_fs, \
        out_fs, \
        act_funs, \
        pool_sizes, \
        kernel_sizes, \
    *args, **kwargs):

    block = [
        conv_block(\
            in_f, \
            out_f, \
            act_fun, \
            pool_size, \
            kernel_size, \
        padding="same", *args, **kwargs)
        # 
        for in_f, out_f, act_fun, pool_size, kernel_size in zip(
            [*in_fs],
            [*out_fs],
            [*act_funs],
            [*pool_sizes],
            [*kernel_sizes],
        )
    ]
    return torch.nn.Sequential(*block)

def dense_block(in_f, out_f, act_fun, *args, **kwargs):
    if act_fun == 'linear':
        return torch.nn.Sequential(
            torch.nn.Linear(in_f, out_f, bias=True),
        )
    else:
        return torch.nn.Sequential(
            torch.nn.Linear(in_f, out_f, bias=True),
            getattr(torch.nn, act_fun)(),
        )

def dense_lazy_block(out_f, act_fun, *args, **kwargs):
    if act_fun == 'linear':
        return torch.nn.Sequential(
            torch.nn.LazyLinear(out_f, bias=True),
        )
    else:
        return torch.nn.Sequential(
            torch.nn.LazyLinear(out_f, bias=True),
            getattr(torch.nn, act_fun)(),
        )

def dense_sequence(\
        in_fs, \
        out_fs, \
        act_funs, \
    *args, **kwargs):

    block = []
    for in_f, out_f, act_fun in zip(
            [*in_fs],
            [*out_fs],
            [*act_funs],
        ):
        if in_f is None:
            block.append(dense_lazy_block(out_f, act_fun, *args, **kwargs))
        else:
            block.append(dense_block(in_f, out_f, act_fun, *args, **kwargs))

    return torch.nn.Sequential(*block)

class ConvNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # Check if the config are correct
        assert (
            len(config["cnn_act_funs"])
            == len(config["pool_sizes"])
            == len(config["kernel_sizes"])
            == len(config["filters"])
        )
        assert (
            len(config["dense_act_funs"])
            == len(config["hiddens"])
        )

        # Longitude padding
        if config["circular_padding"]:
            pads = (config["circular_padding"],config["circular_padding"],0,0)
            self.pad_lons = torch.nn.CircularPad2d(pads)
        
        # Convolutional layers
        self.conv = conv_sequence(
            [config["n_inputs"], *config["filters"][:-1]],
            [*config["filters"]],
            [*config["cnn_act_funs"]],
            [*config["pool_sizes"]],
            [*config["kernel_sizes"]],
        )

        # Flattening
        self.flat = nn.Flatten(start_dim=1)
        
        # Dense layers
        self.dense = dense_sequence(
            [None, *config["hiddens"][:-1]],
            [*config["hiddens"]],
            [*config["dense_act_funs"]],
        )

        # Output layer
        self.out = nn.Linear(config["hiddens"][-1], 1, bias=True) 

    def forward(self, x):

        if self.config['circular_padding']:
            x = self.pad_lons(x)

        x = self.conv(x)
        x = self.flat(x)
        x = self.dense(x)
        x = self.out(x)

        return x
    


    def predict(self, dataset=None, dataloader=None, batch_size=128, device="cpu"):

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

        self.to(device)
        self.eval()
        with torch.inference_mode():
            output = None
            for batch_idx, (data, target) in enumerate(dataloader):
                input, target = (
                    data.to(device),
                    target.to(device),
                )

                out = self(input).to("cpu").numpy()
                if output is None:
                    output = out
                else:
                    output = np.concatenate((output, out), axis=0)

        return output
    
    def gradient(self, dataset=None, dataloader=None, batch_size=128, device="cpu"):

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

        self.to(device)
        self.eval()
        grads = None
        for batch_idx, (input, target) in enumerate(dataloader):
            input, target = (
                input.to(device),
                target.to(device),
            )

            self.zero_grad()
            input.requires_grad_(True)
            out = self(input)
            out.backward(torch.ones_like(out))

            if grads is None:
                grads = input.grad.to('cpu').numpy()
            else:
                grads = np.concatenate((grads, input.grad.to('cpu').numpy()), axis=0)

        return grads



class LinearNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # Flattening
        self.flat = nn.Flatten(start_dim=1)

        # Output layer
        self.out = nn.Linear(self.config["input_size"], 1, bias=config['bias'])

    def forward(self, x):

        x = self.flat(x)
        x = self.out(x)

        return x
    


    def predict(self, dataset=None, dataloader=None, batch_size=128, device="cpu"):

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

        self.to(device)
        self.eval()
        with torch.inference_mode():
            output = None
            for batch_idx, (data, target) in enumerate(dataloader):
                input, target = (
                    data.to(device),
                    target.to(device),
                )

                out = self(input).to("cpu").numpy()
                if output is None:
                    output = out
                else:
                    output = np.concatenate((output, out), axis=0)

        return output
    


class GlobalAverageNet(nn.Module):
    def __init__(self, orig_model, config):
        super().__init__()

        self.config = config
        self.orig_model = orig_model

        # Flattening
        self.flat = nn.Flatten(start_dim=1)

        # Output layer
        self.out = nn.Linear(self.config["input_size"], 1, bias=config['bias'])

    def forward(self, x):

        if self.orig_model is not None:
            x = self.orig_model(x)
        x = self.flat(x)
        x = self.out(x)

        return x