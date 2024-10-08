"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: Utility functions.
"""

import cProfile
import os
import pstats
import random as python_random
import sys

import numpy as np
import torch
import yaml
from numpy.random import seed
from sklearn import manifold
from texttable import Texttable
import logging




def set_seed(options):
    """
    Sets the seed for reproducibility in various modules (numpy, torch, etc.).

    Parameters
    ----------
    options : dict
        Dictionary containing seed value under the key "seed".
    """
    seed = options["seed"]
    np.random.seed(seed)
    python_random.seed(seed)
    torch.manual_seed(seed)


def create_dir(dir_path):
    """
    Creates a directory if it does not already exist.

    Parameters
    ----------
    dir_path : str
        The path of the directory to create.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def set_dirs(config):
    """
    Sets up directory structure for loading data, saving results, and logging.

    The directory structure follows the pattern:
        results > experiment > training > task > model
                                      > evaluation > plots > loss

    Parameters
    ----------
    config : dict
        Dictionary containing configuration options including paths and dataset names.
    """
    paths = config["paths"]

    # Data directories
    data_processed_dir = make_dir(paths["data"], "processed")
    data_pdb_dir = make_dir(paths["data"], f"{config['dataset']}_{config['scheme']}")
    config["data_pdb_dir"] = data_pdb_dir
    sabdab_pdb_dir = make_dir(paths["data"], f"sabdab_{config['scheme']}")
    config["sabdab_pdb_dir"] = sabdab_pdb_dir

    # Results directories
    results_dir = make_dir(paths["results"], "")
    results_dir = make_dir(results_dir, config["experiment"])
    training_dir = make_dir(results_dir, "training")
    evaluation_dir = make_dir(results_dir, "evaluation")
    model_mode_dir = make_dir(training_dir, config["task"])
    training_model_dir = make_dir(model_mode_dir, "model")
    training_plot_dir = make_dir(model_mode_dir, "plots")
    training_loss_dir = make_dir(model_mode_dir, "loss")
    config["results_dir"] = results_dir

    if 'antigen' in config and 'antibody' in config:
        evaluation_dir_abag = make_dir(results_dir, f"{config['antigen']}_{config['antibody']}")
        lo_dir = make_dir(evaluation_dir_abag, "lead_optimization")
        denovo_dir = make_dir(evaluation_dir_abag, "denovo")
        docking_dir = make_dir(evaluation_dir_abag, "docking")
        fixbb_dir = make_dir(evaluation_dir_abag, "fixbb")
        docking_dir = make_dir(docking_dir, config['antigen'])
        docked_structures_dir = make_dir(docking_dir, "docked_structures")
        scheme_dir = make_dir(docking_dir, config['scheme'])
        scores_dir = make_dir(evaluation_dir_abag, "scores")
        preds_dir = make_dir(evaluation_dir_abag, "predictions")

        config.update({
            "denovo_dir": denovo_dir,
            "lo_dir": lo_dir,
            "docking_dir": docking_dir,
            "scores_dir": scores_dir,
            "preds_dir": preds_dir,
            "scheme_dir": scheme_dir,
            "docked_structures_dir": docked_structures_dir,
            "fixbb_dir": fixbb_dir
        })

    print("Directories are set.")


def make_dir(directory_path, new_folder_name):
    """
    Creates a directory with the specified folder name if it does not exist.

    Parameters
    ----------
    directory_path : str
        Parent directory path.
    new_folder_name : str
        Name of the new folder to create.

    Returns
    -------
    str
        Full path of the created directory.
    """
    directory_path = os.path.join(directory_path, new_folder_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path


def get_runtime_and_model_config(args):
    """
    Loads runtime configuration from a YAML file.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.

    Returns
    -------
    dict
        Configuration dictionary loaded from the YAML file.
    """
    try:
        with open(f"./config/{args.dataset}.yaml", "r") as file:
            config = yaml.safe_load(file)
    except Exception as e:
        sys.exit("Error reading runtime config file")

    config["dataset"] = args.dataset
    return config


def update_config_with_model_dims(data_loader, config):
    """
    Updates configuration with the dimension of input features.

    Parameters
    ----------
    data_loader : DataLoader
        DataLoader object for the training dataset.
    config : dict
        Configuration dictionary to be updated.

    Returns
    -------
    dict
        Updated configuration dictionary.
    """
    x, _ = next(iter(data_loader.train_loader))
    dim = x.shape[-1]
    config["dims"].insert(0, dim)
    return config


def run_with_profiler(main_fn, config):
    """
    Runs the main function with a profiler to measure time spent on each step.

    Parameters
    ----------
    main_fn : function
        Main function to run with profiling.
    config : dict
        Configuration dictionary.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    main_fn(config)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('ncalls')
    stats.print_stats()


def tsne(latent):
    """
    Reduces the dimensionality of the embeddings using t-SNE.

    Parameters
    ----------
    latent : np.ndarray
        Input embeddings to reduce dimensionality.

    Returns
    -------
    np.ndarray
        Reduced 2D embeddings.
    """
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    return tsne.fit_transform(latent)


def print_config(args):
    """
    Prints out the configuration settings in a tabular format.

    Parameters
    ----------
    args : dict or argparse.Namespace
        Configuration or arguments to print.
    """
    if not isinstance(args, dict):
        args = vars(args)

    keys = sorted(args.keys())
    table = Texttable()
    table.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(table.draw())


def nan_to_none_or_empty_str(val, string=True):
    """
    Converts NaN values to None or an empty string based on the input type.

    Parameters
    ----------
    val : any
        Input value.
    string : bool, optional
        Whether to convert to an empty string if `val` is NaN, by default True.

    Returns
    -------
    any
        None or an empty string if `val` is NaN, otherwise returns `val`.
    """
    if val != val or not val:
        return '' if string else None
    return val


def split_delimiter_pipe2str(val):
    """
    Splits a string by pipe ('|') and trims whitespace from the resulting parts.

    Parameters
    ----------
    val : str
        Input string to split.

    Returns
    -------
    list
        List of split and trimmed strings.
    """
    if not val:
        return []
    return [s.strip() for s in val.split('|')]


def parse_resolution(val):
    """
    Parses the resolution value from a string.

    Parameters
    ----------
    val : str or None
        Input resolution value.

    Returns
    -------
    float or None
        Parsed resolution as a float, or None if input is invalid.
    """
    if val in {'NOT', '', None} or val != val:
        return None

    if isinstance(val, str) and ',' in val:
        return float(val.split(',')[0].strip())

    return float(val)


def get_logger(name, log_dir=None):
    """
    Configures and returns a logger for logging messages to console and file.

    Parameters
    ----------
    name : str
        Logger name.
    log_dir : str, optional
        Directory to save log files, by default None.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
