"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: Training script.
"""

# Standard library imports
import os
import traceback
from datetime import date, timedelta

# Third-party library imports
import yaml
import torch
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy

# Custom module imports
from src.model import DiffAbXLWrapper
from utils.load_data import AbLoader
from utils.arguments import get_arguments, get_config, print_config_summary
from utils.utils import set_dirs

# Set the NCCL blocking wait environment variable to avoid deadlocks in multi-GPU training
os.environ["NCCL_BLOCKING_WAIT"] = "1"


def train(config, data_loader):
    """
    Trains the DiffAbXL model using the provided configuration and data loader.

    Args:
        config (dict): A dictionary containing configuration options and arguments.
        data_loader (AbLoader): Data loader containing training and validation datasets.

    """
    # Instantiate the model
    model = DiffAbXLWrapper(config)

    # Monitor learning rate during training
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [lr_monitor]

    # Model checkpointing configuration
    filename = config["model_name"]
    checkpoint_callback = ModelCheckpoint(
        dirpath=model._model_path,
        filename=filename,
        save_top_k=1,
        verbose=True,
        every_n_epochs=config["model_save_freq"],
    )
    callbacks.append(checkpoint_callback)

    # Set up logging (Wandb or CSV depending on config)
    today = date.today()
    log_name = f"{filename}_{today.strftime('%Y-%m-%d')}"
    logger = (
        WandbLogger(project="diff_allCDRs", name=log_name, log_model=False)
        if config["wandb"]
        else CSVLogger(save_dir=config["results_dir"], name=log_name)
    )

    # Get training and validation data loaders
    train_loader = data_loader.train_loader
    validation_loader = data_loader.validation_loader

    # Configure the trainer
    trainer = L.Trainer(
        devices=config['num_gpus'],
        accelerator="gpu",
        strategy=DDPStrategy(timeout=timedelta(seconds=15400), find_unused_parameters=True),
        precision=32,
        max_epochs=config["epochs"],
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        val_check_interval=config["val_check_interval"],
        log_every_n_steps=config["log_every_n_steps"],
    )

    # Start training
    trainer.fit(model, train_loader, validation_loader)

    # Save the config file for future reference
    config_path = f"{model._results_path}/config.yml"
    with open(config_path, "w") as config_file:
        yaml.dump(config, config_file, default_flow_style=False)

    print("Training completed successfully.")


def main(config):
    """
    Main entry point for the training process. Sets up necessary directories, loads data, 
    and initiates the training process.

    Args:
        config (dict): A dictionary containing configuration options and arguments.

    """
    # Ensure necessary directories are set up
    set_dirs(config)

    # Load the data for the specified dataset
    ds_loader = AbLoader(config, dataset_name=config["dataset"])

    # Proceed with training
    train(config, ds_loader)


if __name__ == "__main__":
    # Parse command-line arguments
    args = get_arguments()

    # Load the configuration file
    config = get_config(args)

    # Experiment name --- can be changed, default is the name of the dataset
    config["experiment"] = config["dataset"]

    # Print configuration summary for verification
    print_config_summary(config, args)

    # Run the main training function with error handling
    try:
        main(config)
    except Exception as e:
        print(traceback.format_exc())
