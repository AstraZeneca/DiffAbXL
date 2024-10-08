"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: DiffAbXL wrapper class.
"""
import gc
import os
import torch
import numpy as np
import pandas as pd
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from utils.model_utils import DiffAbXL
from utils.utils import set_seed, set_dirs
from utils.utils_diff import move2device
from utils.loss_functions import sum_weighted_losses
import lightning as L


torch.autograd.set_detect_anomaly(True)


class DiffAbXLWrapper(L.LightningModule):
    """
    Model: Trains DiffAbXL. This wrapper class manages the training and validation
    steps, optimizer configuration, and memory management for DiffAbXL models.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model parameters and settings.
    """

    def __init__(self, config):
        """
        Initialize the DiffAbXLWrapper model wrapper class.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing model parameters and settings.
        """
        super().__init__()
        self.config = config
        self.model_dict = {}
        self.loss = {
            "tloss_b": [],
            "seqloss_b": [],
            "rotloss_b": [],
            "posloss_b": [],
            "grad": [],
            "vtloss_b": [],
            "vseqloss_b": [],
            "vrotloss_b": [],
            "vposloss_b": []
        }
        self.automatic_optimization = True  # Enable automatic optimization in Lightning

        # Set random seed and directories
        set_seed(self.config)
        set_dirs(self.config)

        # Initialize model and paths
        self._set_paths()
        self.set_diffabxl()
        self.print_model_summary()

    def set_diffabxl(self):
        """Sets up the DiffAbXL model and moves it to the appropriate device."""
        self.encoder = DiffAbXL(self.config)
        self.encoder.to(self.device)
        self.model_dict.update({"encoder": self.encoder})

    def clean_up_memory(self, losses):
        """
        Frees memory by deleting loss variables and running garbage collection.

        Parameters
        ----------
        losses : list
            List of loss tensors to delete and clean up.
        """
        for loss in losses:
            del loss
        gc.collect()

    def training_step(self, batch, batch_idx):
        """
        Executes a single training step. It computes the loss, performs a backward pass,
        and updates the model parameters.

        Parameters
        ----------
        batch : dict
            Input data batch.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            Computed loss for the training step.
        """
        data = move2device(batch, self.config['device'])

        # Forward pass
        loss_all = self.encoder(data)

        # Compute the overall loss
        enc_loss = sum_weighted_losses(loss_all, self.config['loss_weights'])

        # Log the individual losses
        self.loss["tloss_b"].append((loss_all['seq'].sum().item() + loss_all['rot'].sum().item() + loss_all['pos'].sum().item()))
        self.loss["seqloss_b"].append(loss_all['seq'].sum().item())
        self.loss["rotloss_b"].append(loss_all['rot'].sum().item())
        self.loss["posloss_b"].append(loss_all['pos'].sum().item())

        self.log("Tot", self.loss['tloss_b'][-1], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Pos", self.loss['posloss_b'][-1], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Rot", self.loss['rotloss_b'][-1], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Seq", self.loss['seqloss_b'][-1], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return enc_loss

    def validation_step(self, batch, batch_idx):
        """
        Executes a single validation step. It computes and records the validation losses.

        Parameters
        ----------
        batch : dict
            Input data batch.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            The computed loss for the validation step.
        """
        loss, loss_all = self.step(batch)

        # Save validation losses
        self.loss["vtloss_b"].append((loss_all['seq'].sum().item() + loss_all['rot'].sum().item() + loss_all['pos'].sum().item()))
        self.loss["vseqloss_b"].append(loss_all['seq'].sum().item())
        self.loss["vrotloss_b"].append(loss_all['rot'].sum().item())
        self.loss["vposloss_b"].append(loss_all['pos'].sum().item())

        return loss

    def step(self, batch):
        """
        Computes the loss for a given data batch.

        Parameters
        ----------
        batch : dict
            Input data batch.

        Returns
        -------
        tuple
            Tuple of total loss and individual loss components.
        """
        data = move2device(batch, self.config['device'])
        loss_all = self.encoder(data)
        loss = sum_weighted_losses(loss_all, self.config['loss_weights'])

        return loss, loss_all

    def print_model_summary(self):
        """
        Prints a summary of the model architecture and its parameters.
        """
        description = f"{40 * '-'}Summary of the models:{40 * '-'}\n"
        description += f"{self.encoder}\n"
        description += f"{100 * '*'}\n"
        self.config["total_params"] = str(round(sum(p.numel() for _, model in self.model_dict.items() for p in model.parameters()) / 1e6, 2)) + ' Million'
        description += f"Total number of trainable parameters: {self.config['total_params']}\n"
        description += f"{100 * '*'}\n"
        print(description)

    def _set_paths(self):
        """
        Sets up the directory paths for saving results, models, plots, and losses.
        """
        self._results_path = os.path.join(self.config["paths"]["results"], self.config["experiment"])
        self._model_path = os.path.join(self._results_path, "training", self.config["task"], "model")
        self._plots_path = os.path.join(self._results_path, "training", self.config["task"], "plots")
        self._loss_path = os.path.join(self._results_path, "training", self.config["task"], "loss")

    def configure_optimizers(self):
        """
        Sets up the AdamW optimizer and learning rate scheduler.

        Returns
        -------
        dict
            Dictionary containing optimizer and learning rate scheduler configuration.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config["learning_rate"], betas=(0.9, 0.999), weight_decay=0)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.8, patience=self.config["patience"], min_lr=self.config["min_lr"])
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'Tot_epoch'}
