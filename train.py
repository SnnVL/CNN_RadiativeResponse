"""
This script trains a model for peak temperature prediction using climate data. It loads the necessary modules and libraries, sets up the data, builds and trains the model, computes metrics, and saves the model and metrics.

Usage:
    python train.py <expname> [--overwrite]

Arguments:
    expname (str): The experiment name to specify the config file, e.g. exp101
    --overwrite: Set this flag to overwrite existing models. If not set, the script will skip training if a model with the same name already exists.

Example:
    python train.py exp101
"""

import sys
import xarray as xr
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import torchinfo
import importlib as imp
import pandas as pd
import warnings
import argparse
import os.path

from data_loader.data_generator import ClimateData
from trainer.trainer import Trainer
from model.model import ConvNet, LinearNet
from utils import utils
import model.loss as module_loss
import model.metric as module_metric
import data_loader.data_loaders as data_loader
from utils.DIRECTORIES import MODEL_DIRECTORY, SHAPE_DIRECTORY

warnings.filterwarnings("ignore")
torch.set_warn_always(False)

print(f"python version = {sys.version}")
print(f"numpy version = {np.__version__}")
print(f"xarray version = {xr.__version__}")
print(f"pytorch version = {torch.__version__}")

# --------------------------------------------------------
OVERWRITE = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "expname", help="experiment name to specify the config file, e.g. exp101"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Set this flag to overwrite existing models"
    )
    args = parser.parse_args()
    config = utils.get_config(args.expname)
    
    # Set OVERWRITE based on the flag
    OVERWRITE = args.overwrite

    # Loop through random seeds
    for seed in config["seed_list"]:

        # make model initialization deterministic
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

        model_name = utils.get_model_name(config["expname"], seed)
        if (os.path.isfile(MODEL_DIRECTORY + model_name + ".pt")) & (not OVERWRITE):
            continue
        print("___________________")
        print(model_name)

        # Get the Data
        print("___________________")
        print("Get the data.")
        data = ClimateData(
            config["datamaker"],
            expname=config["expname"],
            seed=seed,
            verbose=True,
        )

        if config["datamaker"]["map_output"]:
            trainset = data_loader.MapToMapData(data.d_train)
            valset = data_loader.MapToMapData(data.d_val)
            testset = data_loader.MapToMapData(data.d_test)
        else:
            trainset = data_loader.MapToValueData(data.d_train)
            valset = data_loader.MapToValueData(data.d_val)
            testset = data_loader.MapToValueData(data.d_test)

        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=config["datamaker"]["batch_size"],
            shuffle=True,
            drop_last=False,
        )
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=config["datamaker"]["batch_size"],
            shuffle=False,
            drop_last=False,
        )

        # Setup the Model
        print("___________________")
        print("Building and training the model.")

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
        
        # Load pretrained model if specified
        if config["datamaker"]["load_pretrained"]:
            pretrained_model_name = utils.get_model_name(config["datamaker"]["load_pretrained"], config["datamaker"]["pretrained_seed"])
            model = utils.load_torch_model(model, pretrained_model_name + ".pt")

        # Setup device, optimizer, criterion, and scheduler
        device = utils.prepare_device(config["device"])
        optimizer = getattr(torch.optim, config["optimizer"]["type"])(
            model.parameters(), **config["optimizer"]["args"]
        )
        if config["datamaker"]["label_mask"]:
            mask = xr.open_dataarray(SHAPE_DIRECTORY + config["datamaker"]["label_mask"]).values
        else:
            mask = None
        if config["criterion"] == 'mse_sphere':
            criterion = module_loss.mse_sphere(data.label_lat,data.label_lon, device, mask=mask)
        else:
            criterion = getattr(module_loss, config["criterion"])
        if "scheduler" in config.keys():
            scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"]["type"])(
                optimizer, **config["scheduler"]["args"]
            )
        else:
            scheduler = None

        metric_funcs = [getattr(module_metric, met) for met in config["metrics"]]

        # Build the trainer
        trainer = Trainer(
            model,
            criterion,
            metric_funcs,
            optimizer,
            scheduler,
            max_epochs=config["trainer"]["max_epochs"],
            data_loader=train_loader,
            validation_data_loader=val_loader,
            device=device,
            config=config,
            do_validation=config["trainer"]["do_validation"],
        )

        # Visualize the model
        print(torchinfo.summary(
            model,
            trainset.input[: config["datamaker"]["batch_size"]].shape,
            verbose=0,
            col_names=("input_size", "output_size", "num_params"),
        ))

        # Train the Model
        model.to(device)
        trainer.fit()
        model.eval()

        # Save the Pytorch Model
        utils.save_torch_model(model, model_name + ".pt")
        print("Completed " + model_name)

        print(trainer.log.history.keys())
        plt.figure(figsize=(5*(len(config["metrics"])+1), 4))
        for i, m in enumerate(("loss", *config["metrics"])):
            plt.subplot(1, len(config["metrics"])+1, i + 1)
            plt.plot(trainer.log.history["epoch"], trainer.log.history[m], label=m)
            plt.plot(
                trainer.log.history["epoch"], trainer.log.history["val_" + m], label="val_" + m
            )
            if config["trainer"]["do_validation"]:
                plt.axvline(
                    x=trainer.early_stopper.best_epoch, linestyle="--", color="k", linewidth=0.75
                )
            plt.title(m)
            plt.legend()
        plt.tight_layout()
        plt.savefig(MODEL_DIRECTORY + model_name + '_training.png')