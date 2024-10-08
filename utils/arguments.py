"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: - Collects arguments from command line, and loads configuration from the yaml files.
             - Prints a summary of all options and arguments.
"""

import os
from argparse import ArgumentParser
import sys
import torch
from utils.utils import get_runtime_and_model_config, print_config
import sys
import os
import torch
from argparse import ArgumentParser

class ArgParser(ArgumentParser):
    """
    Inherits from ArgumentParser to provide a more user-friendly error handling mechanism.
    If an error occurs, it prints the help message and exits the program.
    """
    def error(self, message):
        """
        Overrides the default error behavior to print a custom error message and show the help prompt.

        Parameters
        ----------
        message : str
            The error message to display.
        """
        sys.stderr.write(f'error: {message}\n')
        self.print_help()
        sys.exit(2)


def get_arguments():
    """
    Retrieves command line arguments using ArgParser.

    Returns
    -------
    Namespace
        Parsed command line arguments.
    """
    parser = ArgParser()

    # Dataset to use (must have a corresponding config file)
    parser.add_argument(
        "-d", "--dataset", type=str, default="sabdab",
        help="Name of the dataset to use. It should have a config file with the same name."
    )

    # Epoch for loading saved models
    parser.add_argument(
        "-e", "--epoch", type=int, default=None,
        help="Defines the epoch when loading the model, if the model is saved at specific epochs."
    )

    # Batch size
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=None,
        help="Defines batch size. If None, use batch size defined in the config file."
    )

    # GPU usage
    parser.add_argument(
        "-g", "--gpu", dest="gpu", action="store_true",
        help="Assigns GPU as the device, assuming that GPU is available."
    )
    parser.add_argument(
        "-ng", "--no_gpu", dest="gpu", action="store_false",
        help="Assigns CPU as the device."
    )
    parser.set_defaults(gpu=True)

    # Fine-tuning option
    parser.add_argument(
        "-ft", "--fine_tune", dest="fine_tune", action="store_true",
        help="Used to fine-tune the model."
    )
    parser.set_defaults(fine_tune=False)

    # Device number for GPU (e.g., "cuda:0")
    parser.add_argument(
        "-dn", "--device_number", type=str, default="0",
        help="Defines which GPU to use. Default is 0."
    )

    # Experiment number for MLFlow
    parser.add_argument(
        "-ex", "--experiment", type=int, default=1,
        help="Used as a suffix for MLFlow experiments, if MLFlow is enabled."
    )

    # Antibody-antigen docking options
    parser.add_argument(
        "--antigen", type=str, default="./data/examples/Omicron_RBD.pdb",
        help="Path to the antigen structure (PDB file)."
    )
    parser.add_argument(
        "--antibody", type=str, default="./data/examples/3QHF_Fv.pdb",
        help="Path to the antibody structure (PDB file)."
    )
    parser.add_argument(
        "-nd", "--num_docks", type=int, default=10,
        help="Number of docking attempts."
    )
    parser.add_argument(
        "--heavy", type=str, default="H",
        help="Chain ID of the heavy chain."
    )
    parser.add_argument(
        "--light", type=str, default="L",
        help="Chain ID of the light chain."
    )

    return parser.parse_args()


def get_config(args):
    """
    Loads the configuration settings from YAML files and incorporates command line arguments.

    Parameters
    ----------
    args : Namespace
        Command line arguments parsed from `get_arguments()`.

    Returns
    -------
    dict
        Configuration dictionary combining runtime settings and command line arguments.
    """
    config = get_runtime_and_model_config(args)
    
    # Device setup: GPU or CPU
    config["device"] = torch.device(
        f'cuda:{args.device_number}' if torch.cuda.is_available() and args.gpu else 'cpu'
    )

    # Epoch override if specified
    config["epoch"] = args.epoch

    # Antibody settings
    config["heavy"] = args.heavy
    config["light"] = args.light
    config["dataset"] = args.dataset

    # System configurations
    config["num_workers"] = os.cpu_count()

    # Batch size override if specified
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    return config


def print_config_summary(config, args=None):
    """
    Prints a summary of the current configuration and command line arguments (if provided).

    Parameters
    ----------
    config : dict
        The configuration dictionary to print.
    args : Namespace, optional
        Parsed command line arguments, if available.
    """
    print("=" * 100)
    print("Here is the configuration being used:\n")
    print_config(config)
    print("=" * 100)
    
    if args is not None:
        print("Arguments being used:\n")
        print_config(args)
        print("=" * 100)