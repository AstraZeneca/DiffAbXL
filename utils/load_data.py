"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: A library for data loaders.
"""

import lmdb
import os
import pickle
import random
from itertools import chain

import pandas as pd
import torch
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from tqdm.auto import tqdm

from utils.utils import set_seed
from utils.protein_constants import AALib
from utils.transformations import MaskCDRs, MaskAntibody, MergeChains, PatchAroundAnchor, RemoveAntigen


class TransformComplex:
    """
    Composes a set of transformations based on the given configuration.

    Parameters
    ----------
    config : dict
        Dictionary containing transformation configuration options.
    """
    def __init__(self, config):
        super().__init__()
        transform_dict = {
            'mask_cdrs': MaskCDRs(config),
            'mask_antibody': MaskAntibody(config),
            'merge_chains': MergeChains(config),
            'patch_around_anchor': PatchAroundAnchor(config, patch_size=config["patch_size"], antigen_size=config["patch_size"]),
            'remove_antigen': RemoveAntigen,
        }
        # Compose selected transformations
        list_of_transforms = [transform_dict[d] for d in config['transform']]
        self.composed_transforms = Compose(list_of_transforms)


class PaddingCollate:
    """
    Pads sequences and collates the data for batching.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing options and arguments.
    training : bool, optional
        Whether the model is in training mode, by default True.
    padding_token : int or str, optional
        Token to use for padding sequences, by default AALib.PAD.

    Attributes
    ----------
    pad_values : dict
        A dictionary specifying the padding values for different variables:
        - 'aa': Uses `padding_token` (typically AALib.PAD).
        - 'chain_id': Pads with a space (' ').
        - 'icode': Pads with a space (' ').
        - 'structure_type': Pads with a space (' ').
    """
    def __init__(self, config, training=True, padding_token=AALib.PAD):
        super().__init__()
        self.config = config
        self.training = training
        self.pad_values = {
            'aa': padding_token,
            'chain_id': ' ',
            'icode': ' ',
            'structure_type': ' ',
        }
        # Fields that do not require padding
        self.no_padding = {'origin'}

    def __call__(self, data_list):
        """
        Pads each data sample in `data_list` to the maximum length and collates them.

        Parameters
        ----------
        data_list : list
            List of data samples to be padded and collated.

        Returns
        -------
        dict
            Padded and collated batch of data.
        """
        max_length = self.config["max_length"]
        keys = self._get_common_keys(data_list)
        data_list_padded = []

        for data in data_list:
            data_padded = {}

            for k, v in data.items():
                if k in keys and v is not None:
                    value = v
                    if k not in self.no_padding:
                        # Pad sequences to the max_length if necessary
                        if (isinstance(v, torch.Tensor) and v.size(0) <= max_length) or (isinstance(v, list) and len(v) <= max_length):
                            value = self._pad_last(v, max_length, value=self._get_pad_value(k))
                        else:
                            value = v

                    data_padded.update({k: value})

            if data_padded:
                # Create padding mask using the hardcoded length reference key ('aa')
                data_padded['residue_mask'] = self._get_pad_mask(data['aa'].size(0), max_length)
                data_list_padded.append(data_padded)

        try:
            final_data = default_collate(data_list_padded)
            return final_data
        except Exception as e:
            print(e)

    @staticmethod
    def _pad_last(x, n, value=0):
        """
        Pads the sequence `x` to length `n`.

        Parameters
        ----------
        x : torch.Tensor or list
            Input sequence to pad.
        n : int
            Length to pad to.
        value : int, optional
            Value to use for padding, by default 0.

        Returns
        -------
        torch.Tensor or list
            Padded sequence.
        """
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x

            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)

        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        return x

    @staticmethod
    def _get_pad_mask(l, n):
        """
        Creates a padding mask indicating which parts of the sequence are padded.

        Parameters
        ----------
        l : int
            Length of the original sequence.
        n : int
            Total padded length.

        Returns
        -------
        torch.Tensor
            Padding mask.
        """
        return torch.cat([torch.ones([l], dtype=torch.bool), torch.zeros([n - l], dtype=torch.bool)], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        """
        Gets the common keys across a list of dictionaries.

        Parameters
        ----------
        list_of_dict : list
            List of dictionaries to extract common keys from.

        Returns
        -------
        set
            Set of common keys.
        """
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys

    def _get_pad_value(self, key):
        """
        Returns the padding value for a specific key.

        Parameters
        ----------
        key : str
            Key for which to retrieve the padding value.

        Returns
        -------
        int or str
            Padding value for the key.
        """
        return self.pad_values.get(key, 0)


class AbLoader:
    """
    Data loader for training, validation, and test datasets.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing options and arguments.
    dataset_name : str
        Name of the dataset to load.
    drop_last : bool, optional
        Whether to drop the last incomplete batch, by default True.
    kwargs : dict, optional
        Additional keyword arguments, by default {}.
    """
    def __init__(self, config, dataset_name, drop_last=True, kwargs={}):
        super().__init__()
        bs = config["batch_size"]
        nw = min(8, config["num_workers"])
        self.config = config

        # Get the datasets
        train_dataset, test_dataset, validation_dataset, rabd_dataset = self.get_dataset(dataset_name)

        # Set the loader for training set
        self.train_loader = DataLoader(train_dataset, batch_size=bs, collate_fn=PaddingCollate(config, padding_token=AALib.PAD), shuffle=True, drop_last=drop_last, num_workers=nw)
        if self.config['load_val_test']:
            # Set the loader for test set
            self.test_loader = DataLoader(test_dataset, batch_size=bs, collate_fn=PaddingCollate(config, padding_token=AALib.PAD, training=False), shuffle=False, drop_last=drop_last, num_workers=nw)
            # Set the loader for validation set
            self.validation_loader = DataLoader(validation_dataset, batch_size=bs, collate_fn=PaddingCollate(config, padding_token=AALib.PAD, training=False), shuffle=False, drop_last=drop_last, num_workers=nw)
            # Set the loader for rabd set
            self.rabd_loader = rabd_dataset

    def get_dataset(self, dataset_name):
        """
        Returns the datasets for training, validation, test, and rabd sets.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to load.

        Returns
        -------
        tuple
            Training, validation, test, and rabd datasets.
        """
        reset = self.config['reset']
        is_transform = self.config['is_transform']
        # Training dataset
        train_dataset = StructureDataset(self.config, split='train', reset=reset, is_transform=is_transform)
        # Validation dataset
        validation_dataset = None if reset else StructureDataset(self.config, split="val", reset=False, is_transform=is_transform)
        # Test dataset
        test_dataset = None if reset else StructureDataset(self.config, split='test', reset=False, is_transform=False)
        # RABD dataset
        rabd_dataset = None if reset else StructureDataset(self.config, split='test', reset=False, is_transform=False)

        return train_dataset, test_dataset, validation_dataset, rabd_dataset


class StructureDataset(Dataset):
    """
    Dataset class for handling tabular data.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing options and arguments.
    split : str, optional
        The data split, either 'train', 'val', or 'test', by default 'train'.
    reset : bool, optional
        Whether to reset the dataset, by default False.
    is_transform : bool, optional
        Whether to apply data transformations, by default False.
    """
    def __init__(self, config, split='train', reset=False, is_transform=False):
        super().__init__()
        transform = TransformComplex(config)

        self.config = config
        self.split = split
        self.reset = reset
        self.is_transform = is_transform
        self.transform = transform.composed_transforms
        self.db_connection = None

        # Data paths
        self.data_path = config["paths"]["data"]
        self.processed_dir = config["data_pdb_dir"]
        self.map_size = 64 * 1024**3  # Maximum size of the DB

        # Load clusters, entries, and split the data
        self._load_clusters()
        self._load_entries()
        self._load_split()

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.split_ids)

    def __getitem__(self, idx):
        """
        Returns the data for a given index.

        Parameters
        ----------
        idx : int
            Index of the data to retrieve.

        Returns
        -------
        dict
            Data for the given index.
        """
        structure_id = self.split_ids[idx]
        return self._get_data_from_idx(structure_id)

    def _get_data_from_idx(self, structure_id):
        """
        Fetches and transforms data for a given structure ID.

        Parameters
        ----------
        structure_id : str
            The structure ID to fetch data for.

        Returns
        -------
        dict
            Transformed or raw data for the structure.
        """
        data = self._get_structure(structure_id)
        return self.transform(data) if self.is_transform else data

    @property
    def structure_data_path(self):
        """Returns the path to the structure LMDB database."""
        return os.path.join(self.processed_dir, 'structures.lmdb')

    def _get_structure(self, db_id):
        """
        Retrieves the structure data from LMDB.

        Parameters
        ----------
        db_id : str
            The database ID for the structure.

        Returns
        -------
        dict
            The structure data.
        """
        # Initialize LMDB connection if not done yet
        if self.db_connection is None:
            self.db_connection = lmdb.open(self.structure_data_path, 
                                           map_size=self.map_size, 
                                           create=False,
                                           subdir=False,
                                           readonly=True,
                                           lock=False,
                                           readahead=False,
                                           meminit=False)

        # Load structure from LMDB
        with self.db_connection.begin() as txn:
            return pickle.loads(txn.get(db_id.encode()))

    def _load_entries(self):
        """
        Loads the entries of the dataset from a pickle file.
        """
        entries_path = os.path.join(self.processed_dir, 'entries_list.pkl')
        with open(entries_path, 'rb') as f:
            self.all_entries = pickle.load(f)

    def _load_clusters(self):
        """
        Loads the cluster information from a TSV file.
        """
        cluster_path = os.path.join(self.processed_dir, "cluster_results.tsv")
        clusters, id_to_cluster = {}, {}

        with open(cluster_path, 'r') as f:
            for line in f.readlines():
                cluster_name, data_id = line.split()
                clusters.setdefault(cluster_name, []).append(data_id)
                id_to_cluster[data_id] = cluster_name

        self.clusters = clusters
        self.id_to_cluster = id_to_cluster

    def _load_split(self):
        """
        Loads the data split (train, val, or test) based on the configuration.
        """
        # Set the random seed
        set_seed(self.config)

        # Get test IDs
        test_ids = [entry_dict['id'] for entry_dict in self.all_entries if entry_dict is not None and entry_dict['entry']['ag_name'] in self.config['test_set']]
        print(f"Number of initial test IDs: {len(test_ids)}")

        # Get RABD PDB IDs
        rabd_df = pd.read_csv(f"{self.config['paths']['data']}/rabd/rabd.csv", header=None, usecols=[0], names=["ids"])
        rabd_ids = rabd_df["ids"].tolist()

        # Include RABD IDs in the test set
        test_ids += [entry_dict['id'] for entry_dict in self.all_entries if entry_dict is not None and entry_dict['id'][:4] in rabd_ids]
        test_ids_prefix = [tid[:4] for tid in test_ids]
        print(f"Number of final test IDs: {len(test_ids)}")

        # Find clusters corresponding to the test IDs
        test_clusters = list(set([self.id_to_cluster[test_id] for test_id in test_ids]))

        # Load SAbDab PDB IDs
        sabdab_df = pd.read_csv(os.path.join(self.data_path, 'sabdab_summary_all.tsv'), sep='\t')
        sabdab_ids = sabdab_df["pdb"].tolist()

        # Exclude RABD IDs from SAbDab
        sabdab_ids = [sid for sid in sabdab_ids if sid not in test_ids_prefix]

        # Get the remaining SAbDab IDs for training
        train_sabdab_ids = [entry_dict['id'] for entry_dict in self.all_entries if entry_dict is not None and entry_dict['id'][:4] in sabdab_ids]

        # Group the training SAbDab IDs into clusters
        train_clusters_sabdab = list(set([self.id_to_cluster[sid] for sid in train_sabdab_ids if sid in self.id_to_cluster]))

        # Select validation clusters
        num_val_cluster_keys = self.config["validation_size"]
        random.shuffle(train_clusters_sabdab)
        val_clusters = train_clusters_sabdab[:num_val_cluster_keys]
        train_clusters_sabdab = train_clusters_sabdab[num_val_cluster_keys:]

        # Shuffle training clusters again
        random.shuffle(train_clusters_sabdab)

        print("test and val clusters ==================")
        print(f"Number of clusters in test: {len(test_clusters)}")
        print(f"Number of clusters in validation: {len(val_clusters)}")
        print(f"Number of clusters in training SAbDab: {len(train_clusters_sabdab)}")

        # Assign structure IDs based on the selected split
        if self.split == "test":
            self.split_ids = list(chain.from_iterable(self.clusters[c_id] for c_id in test_clusters if c_id in self.clusters))
            print(f"Number of structures in the test split: {len(self.split_ids)}")
        elif self.split == "val":
            self.split_ids = list(chain.from_iterable(self.clusters[c_id] for c_id in val_clusters if c_id in self.clusters))
            print(f"Number of structures in the validation split: {len(self.split_ids)}")
        elif self.split == "rabd":
            self.split_ids = rabd_ids
        else:
            self.split_ids = list(chain.from_iterable(self.clusters[c_id] for c_id in train_clusters_sabdab if c_id in self.clusters))
            print(f"Number of structures from SAbDab in the train split: {len(self.split_ids)}")
