"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: A library of transformations applied to antibodies
"""

import copy
import random
import numpy as np
import torch
from typing import List, Optional
from utils.protein_constants import BBHeavyAtom, AALib, resindex_to_ressymb


class MaskCDRs:
    """
    A class to mask the Complementarity-Determining Regions (CDRs) of an antibody.
    It allows masking of specific CDRs, random CDR selection, and the option to perturb CDR lengths.

    Parameters
    ----------
    config : dict
        Configuration dictionary that specifies the CDRs to mask and other settings.
    sampling : bool, optional
        Whether to use sampling mode (default is False).
    """
    
    def __init__(self, config, sampling=False):
        self.config = config
        self.cdrs_to_mask = config["sampling"]["cdrs_to_mask"] if sampling else config["cdrs_to_mask"]
        self.perturb_length = config["perturb_length"]
        
    def __call__(self, structure):
        """
        Apply masking to CDR regions in the provided antibody.

        Parameters
        ----------
        structure : dict
            Antibody structure containing 'heavy' and 'light' chains.

        Returns
        -------
        dict
            Updated structure with masked CDRs.
        """
        data_to_mask = []
        cdrs_to_mask = copy.deepcopy(self.cdrs_to_mask)

        # If no specific CDRs are provided, randomly select one from the heavy or light chain
        if not cdrs_to_mask:
            ab_data = []
            if structure['heavy'] is not None:
                ab_data.append({'heavy': structure['heavy']})
            if structure['light'] is not None:
                ab_data.append({'light': structure['light']})
            data_to_mask.append(random.choice(ab_data))
            
            # Randomly choose CDR for heavy or light chain
            if 'heavy' in data_to_mask[0]:
                cdrs_to_mask = [random.choice([1, 2, 3])]
            else:
                cdrs_to_mask = [random.choice([4, 5, 6])]

        # Handle specific single CDR masking
        elif len(self.cdrs_to_mask) == 1:
            if cdrs_to_mask[0] in [1, 2, 3]:
                if structure['heavy'] is not None:
                    data_to_mask.append({'heavy': structure['heavy']})
                else:
                    data_to_mask.append({'light': structure['light']})
                    cdrs_to_mask[0] += 3  # Convert heavy CDR to light CDR
            elif cdrs_to_mask[0] in [4, 5, 6]:
                if structure['light'] is not None:
                    data_to_mask.append({'light': structure['light']})
                else:
                    data_to_mask.append({'heavy': structure['heavy']})
                    cdrs_to_mask[0] -= 3  # Convert light CDR to heavy CDR
            else:
                raise ValueError("CDR index should be in the range [1, 2, 3] for heavy or [4, 5, 6] for light chains")

        # Handle multiple CDRs to mask
        elif len(cdrs_to_mask) > 1:
            if structure['heavy'] is not None:
                data_to_mask.append({'heavy': structure['heavy']})
            if structure['light'] is not None:
                data_to_mask.append({'light': structure['light']})

        # Apply the mask to the selected chain(s)
        for data in data_to_mask:
            if 'heavy' in data:
                cdrs_to_apply = [cdr_idx for cdr_idx in cdrs_to_mask if cdr_idx in [1, 2, 3]]
                self.mask_chain(data['heavy'], cdrs_to_mask=cdrs_to_apply)
            
            if 'light' in data:
                cdrs_to_apply = [cdr_idx for cdr_idx in cdrs_to_mask if cdr_idx in [4, 5, 6]]
                self.mask_chain(data['light'], cdrs_to_mask=cdrs_to_apply)

        return structure
    
    def mask_chain(self, data, cdrs_to_mask=None):
        """
        Masks the specified CDRs in a given antibody chain.

        Parameters
        ----------
        data : dict
            Dictionary containing chain data, including CDR locations.
        cdrs_to_mask : list of int, optional
            List of CDR indices to mask (default is None, which will select random CDRs).
        """
        cdr_locations = data["cdr_locations"]
        cdr_types = cdr_locations[cdr_locations > 0].unique().tolist()
        
        if cdrs_to_mask is None:
            random.shuffle(cdr_types)
            num_cdrs_to_mask = random.randint(1, len(cdr_types))
            cdrs_to_mask = cdr_types[:num_cdrs_to_mask]
            
        for cdr in cdrs_to_mask:
            self.mask_single_cdr(data, cdr_to_mask=cdr)

    def mask_single_cdr(self, data, cdr_to_mask=3):
        """
        Masks a single CDR in the provided data.

        Parameters
        ----------
        data : dict
            Dictionary containing chain data, including CDR locations.
        cdr_to_mask : int
            Index of the CDR to mask.
        """
        cdr_locations = data["cdr_locations"]
        
        # Check if the CDR exists, otherwise randomly select one
        if cdr_to_mask is None or sum(cdr_locations == cdr_to_mask) == 0:
            cdr_types = cdr_locations[cdr_locations > 0].unique().tolist()
            cdr_to_mask = random.choice(cdr_types)

        # Create the CDR mask and modify its length if needed
        cdr_mask = (cdr_locations == cdr_to_mask)
        cdr_mask = self.change_length(cdr_mask) if self.perturb_length else cdr_mask

        # Identify anchor points
        cdr_first_idx, cdr_last_idx, _, _ = self.get_start_end_index(cdr_mask)
        left_idx = max(0, cdr_first_idx - 1)
        right_idx = min(data['aa'].size(0) - 1, cdr_last_idx + 1)
        anchor_mask = torch.zeros(data['aa'].shape, dtype=torch.bool)
        anchor_mask[left_idx] = True
        anchor_mask[right_idx] = True

        # Update generation and anchor masks
        if 'generation_mask' not in data:
            data['generation_mask'] = cdr_mask
            data['anchor_mask'] = anchor_mask
        else:
            data['generation_mask'] |= cdr_mask
            data['anchor_mask'] |= anchor_mask

    def change_length(self, mask):
        """
        Randomly shrinks or extends a mask to perturb the CDR length.

        Parameters
        ----------
        mask : torch.Tensor
            Tensor representing the mask for a CDR.

        Returns
        -------
        torch.Tensor
            Perturbed mask with adjusted length.
        """
        min_length = self.config["min_length"]
        shorten_by = self.config["shorten_by"]
        extend_by = self.config["extend_by"]
        
        first_index, last_index, cdr_length, seq_length = self.get_start_end_index(mask)
        
        shorten_by = 0 if (cdr_length - 2 * shorten_by) < min_length else shorten_by
        new_first_index = max(0, first_index - random.randint(-shorten_by, extend_by))
        new_last_index = min(last_index + random.randint(-shorten_by, extend_by), seq_length - 1)
        
        new_mask = torch.zeros_like(mask, dtype=torch.bool)
        new_mask[new_first_index:new_last_index + 1] = True
        
        return new_mask

    def get_start_end_index(self, mask):
        """
        Get the start and end indices of the masked CDR region.

        Parameters
        ----------
        mask : torch.Tensor
            Tensor representing the mask.

        Returns
        -------
        tuple
            First index, last index, CDR length, and sequence length.
        """
        seq_length = mask.size(0)
        indexes = torch.arange(0, seq_length)[mask]
        first_index = indexes[0]
        last_index = indexes[-1]
        cdr_length = mask.sum()
        
        return first_index, last_index, cdr_length, seq_length

class MaskAntibody:
    """
    A class for masking and handling antibody.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary containing various settings such as contact distance.
    """
    def __init__(self, config):
        self.config = config

    def mask_ab_chain(self, data):
        """
        Masks the entire antibody chain by setting the generation mask to True for all residues.

        Parameters
        ----------
        data : dict
            Dictionary containing antibody chain information.
        """
        data['generation_mask'] = torch.ones(data['aa'].shape, dtype=torch.bool)

    def add_ab_calpha(self, structure, pos_ab_calpha, chain='heavy'):
        """
        Adds the C-alpha positions of the specified antibody chain to a list.

        Parameters
        ----------
        structure : dict
            The structure containing the antibody chains.
        pos_ab_calpha : list
            List to store the C-alpha positions.
        chain : str, optional
            The chain to process ('heavy' or 'light'), by default 'heavy'.

        Returns
        -------
        list
            Updated list of C-alpha positions.
        """
        if structure[chain] is not None:
            self.mask_ab_chain(structure[chain])
            pos_ab_calpha.append(structure[chain]['pos_heavyatom'][:, BBHeavyAtom.CA])
        return pos_ab_calpha

    def add_ag_calpha(self, structure, pos_ab_calpha):
        """
        Processes antigen C-alpha positions and computes contact regions with the antibody.

        Parameters
        ----------
        structure : dict
            The structure containing antigen information.
        pos_ab_calpha : torch.Tensor
            Tensor of C-alpha positions from the antibody.

        Returns
        -------
        dict
            Updated structure with antigen contact and anchor masks.
        """
        if structure['antigen'] is not None:
            pos_ag_calpha = structure['antigen']['pos_heavyatom'][:, BBHeavyAtom.CA]
            
            # Calculate pairwise distances between antigen and antibody C-alpha atoms
            ag_ab_dist = torch.cdist(pos_ag_calpha, pos_ab_calpha)
            
            # Find nearest antibody distances for each antigen atom
            nn_ab_dist = ag_ab_dist.min(dim=1)[0]
            contact_mask = (nn_ab_dist <= self.config["contact_distance"])
            
            # Ensure at least one contact exists; if not, pick the closest atom
            if contact_mask.sum().item() == 0:
                contact_mask[nn_ab_dist.argmin()] = True
            
            # Randomly select a contact anchor point
            anchor_idx = torch.multinomial(contact_mask.float(), num_samples=1).item()
            anchor_mask = torch.zeros(structure['antigen']['aa'].shape, dtype=torch.bool)
            anchor_mask[anchor_idx] = True
            
            # Update antigen structure with contact and anchor masks
            structure['antigen']['contact_mask'] = contact_mask
            structure['antigen']['anchor_mask'] = anchor_mask
            
        return structure

    def __call__(self, structure):
        """
        Main function to mask antibody chains and update the structure with antigen contact information.

        Parameters
        ----------
        structure : dict
            Antibody and antigen structure.

        Returns
        -------
        dict
            Updated structure with masked chains and antigen information.
        """
        pos_ab_calpha = []

        # Process heavy and light chains
        pos_ab_calpha = self.add_ab_calpha(structure, pos_ab_calpha, chain='heavy')
        pos_ab_calpha = self.add_ab_calpha(structure, pos_ab_calpha, chain='light')
        
        # Concatenate C-alpha positions from both chains
        pos_ab_calpha = torch.cat(pos_ab_calpha, dim=0)
        
        # Update structure with antigen contact information
        return self.add_ag_calpha(structure, pos_ab_calpha)


class MergeChains:
    """
    A class for merging antibody and antigen chains into a unified structure. Handles multiple chains
    (heavy, light, antigen) and combines them into a single data structure.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing fragment types and other settings.
    is_evaluation : bool, optional
        Flag to indicate if evaluation mode is enabled, by default False.
    """
    def __init__(self, config, is_evaluation=False):
        self.config = config
        self.is_evaluation = is_evaluation
        self.heavy_idx = config['fragment_type']['heavy']
        self.light_kappa_idx = config['fragment_type']['light_kappa']
        self.light_lambda_idx = config['fragment_type']['light_lambda']
        self.antigen_idx = config['fragment_type']['antigen']

    def _data_attr(self, data, name):
        """
        Retrieves a specific attribute from a data dictionary. If the attribute does not exist,
        a default value is returned for mask-related attributes.

        Parameters
        ----------
        data : dict
            Dictionary containing chain or fragment data.
        name : str
            Name of the attribute to retrieve.

        Returns
        -------
        torch.Tensor or list
            The requested attribute or a default value if it doesn't exist.
        """
        if name in ['generation_mask', 'anchor_mask'] and name not in data:
            return torch.zeros(data['aa'].shape, dtype=torch.bool)
        else:
            return data[name]

    def add_fragment(self, fragment, fname='heavy', fidx=1):
        """
        Adds a fragment (e.g., heavy, light, antigen) to the structure, setting its fragment type and
        processing special cases like sequence discrepancies.

        Parameters
        ----------
        fragment : dict
            Fragment data (e.g., heavy, light, or antigen chain).
        fname : str, optional
            Name of the fragment (default is 'heavy').
        fidx : int, optional
            Index representing the fragment type (default is 1).
        """
        if fragment is not None:
            fragment['fragment_type'] = torch.full_like(fragment['aa'], fill_value=fidx)

            # Antigen-specific: initialize CDR locations
            if fname == "antigen":
                fragment['cdr_locations'] = torch.zeros_like(fragment['aa'])

            # Add fragment to the list
            self.fragment_list.append(fragment)

    def __call__(self, structure):
        """
        Merges antibody (heavy and light) and antigen chains into a unified structure, ensuring proper indexing
        and handling of fragment-specific attributes.

        Parameters
        ----------
        structure : dict
            Structure containing heavy, light, and antigen chains.

        Returns
        -------
        dict
            Merged data dictionary containing the combined structure.
        """
        if 'light_ctype' not in structure or structure['light_ctype'] is None:
            structure['light_ctype'] = 'U'  

        self.fragment_list = []

        list_props = {
            'chain_id': [],
            'icode': [],
        }

        tensor_props = {
            'resseq': [],
            'res_nb': [],
            'aa': [],
            'pos_heavyatom': [],
            'mask_heavyatom': [],
            'generation_mask': [],
            'cdr_locations': [],
            'anchor_mask': [],
            'fragment_type': [],
        }

        # Add heavy and light chains to the structure list
        if structure['heavy'] is not None:
            self.add_fragment(structure['heavy'], fname='heavy', fidx=self.heavy_idx)

        if structure['light'] is not None:
            fidx = (
                self.light_kappa_idx if structure['light_ctype'] == 'K'
                else self.light_lambda_idx if structure['light_ctype'] == 'L'
                else self.config["fragment_type"]["unknown_light"]
            )
            self.add_fragment(structure['light'], fname='light', fidx=fidx)

        # Add antigen to the structure list if present
        if structure['antigen'] is not None:
            self.add_fragment(structure['antigen'], fname='antigen', fidx=self.antigen_idx)

        # Add properties of structures into lists
        for fragment in self.fragment_list:
            
            for k in list_props.keys():
                list_props[k].append(self._data_attr(fragment, k))
            
            for k in tensor_props.keys():
                tensor_props[k].append(self._data_attr(fragment, k))

        # Merge fragment properties into unified lists/tensors
        list_props = {k: sum(v, start=[]) for k, v in list_props.items()}
        tensor_props = {k: torch.cat(v, dim=0) for k, v in tensor_props.items()}

        # Return the combined data dictionary
        data_dict = {**list_props, **tensor_props}
        data_dict['structure_type'] = structure['structure_type']

        return data_dict


class PatchAroundAnchor:
    """
    A class that selects a patch of residues around anchor points in antibody-antigen complexes. The patch 
    can include both antibody and antigen residues based on proximity to the anchor points.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing fragment types and other settings.
    patch_size : int, optional
        The size of the patch to extract, by default 200.
    antigen_size : int, optional
        The maximum size of the antigen patch, by default 200.
    is_training : bool, optional
        Whether the model is in training mode, by default True.
    """
    def __init__(self, config, patch_size=200, antigen_size=200, is_training=True):
        self.config = config
        self.is_training = is_training
        self.patch_size = patch_size
        self.antigen_size = antigen_size
        self.antigen_idx = config['fragment_type']['antigen']
        
    def __call__(self, data):
        """
        Selects a patch around the anchor points and antigen contact regions.

        Parameters
        ----------
        data : dict
            A dictionary containing structure and fragment information for antibody-antigen complexes.

        Returns
        -------
        dict
            The structure data with selected patch regions and centered coordinates.
        """
        anchor_mask = data["anchor_mask"]
        generation_mask = data['generation_mask']
                
        # Get anchor points and CDR positions
        anchor_points = data['pos_heavyatom'][anchor_mask, BBHeavyAtom.CA]
        cdr_points = data['pos_heavyatom'][generation_mask, BBHeavyAtom.CA]
        
        # Determine which residues are part of the antigen or antibody
        antigen_mask = (data['fragment_type'] == self.antigen_idx)
        antibody_mask = ~antigen_mask
        
        # If no anchor point is provided, use the entire antibody fragment
        if anchor_mask.sum().item() == 0:
            antibody = self._filter_data(data, antibody_mask)
            origin = antibody['pos_heavyatom'][:, BBHeavyAtom.CA].mean(dim=0)
            antibody = self._center(antibody, origin)
            return antibody
        
        pos_calpha = data['pos_heavyatom'][:, BBHeavyAtom.CA]

        if 'A' in data["structure_type"] and not self.config["antibody_only"]:
            # Compute distances to the anchor points
            dist_to_anchors = torch.cdist(pos_calpha, anchor_points).min(dim=1)[0]
            
            # Get the closest points to the anchor for both antibody and antigen
            initial_patch_idxs = torch.topk(dist_to_anchors, k=min(self.patch_size, dist_to_anchors.size(0)), largest=False)[1]
            dist_to_anchors_antigen = dist_to_anchors.masked_fill(mask=antibody_mask, value=float('inf'))
            antigen_patch_idxs = torch.topk(dist_to_anchors_antigen, k=min(self.antigen_size, antigen_mask.sum().item()), largest=False)[1]
            
            # Create a patch mask that includes both anchor and antigen points
            patch_mask = torch.logical_or(generation_mask, anchor_mask)
            patch_mask[initial_patch_idxs] = True
            patch_mask[antigen_patch_idxs] = True
                
        else:
            # Compute distances to the anchor points (antibody only)
            dist_to_anchors = torch.cdist(pos_calpha, anchor_points).min(dim=1)[0]
            dist_to_anchors_antibody = dist_to_anchors.masked_fill(mask=~antibody_mask, value=float('inf'))
            initial_patch_idxs = torch.topk(dist_to_anchors_antibody, k=min(self.patch_size, dist_to_anchors_antibody.size(0)), largest=False)[1]
            
            # Create a patch mask for the antibody
            patch_mask = torch.logical_or(generation_mask, anchor_mask)
            patch_mask[initial_patch_idxs] = True
            
        # Extract the patch and center the coordinates
        data_patch = self._filter_data(data, patch_mask)
        data_patch = self._center(data_patch, origin=anchor_points.mean(dim=0))

        # Store patch indexes relative to the original sequence length
        patch_idxs = torch.arange(0, patch_mask.shape[0])[patch_mask]
        data_patch['patch_idxs'] = patch_idxs
        
        return data_patch

    def _center(self, data, origin):
        """
        Centers the coordinates of the structure around the origin.

        Parameters
        ----------
        data : dict
            A dictionary containing structure data.
        origin : torch.Tensor
            A tensor representing the new origin for the structure coordinates.

        Returns
        -------
        dict
            Updated structure data with centered coordinates.
        """
        data['origin'] = origin
        data['pos_heavyatom'] = data['pos_heavyatom'] - origin.view(1, 1, 3)
        data['pos_heavyatom'] *= data['mask_heavyatom'][:, :, None]
        return data

    def _filter_data(self, data, mask):
        """
        Filters data based on a given mask.

        Parameters
        ----------
        data : dict
            Structure data to be filtered.
        mask : torch.Tensor
            Boolean mask indicating which parts of the data to keep.

        Returns
        -------
        dict
            Filtered structure data.
        """
        return {k: self._filter(v, mask) for k, v in data.items()}

    def _filter(self, v, mask):
        """
        Filters a tensor or list based on a given mask.

        Parameters
        ----------
        v : torch.Tensor or list
            Data to filter.
        mask : torch.Tensor
            Boolean mask for filtering.

        Returns
        -------
        torch.Tensor or list
            Filtered data.
        """
        if isinstance(v, torch.Tensor) and v.size(0) == mask.size(0):
            return v[mask]
        elif isinstance(v, list) and len(v) == mask.size(0):
            return [v[i] for i, b in enumerate(mask) if b]
        else:
            return v


class RemoveAntigen:
    """
    Removes the antigen from the structure, useful for antibody-only tasks.
    """
    def __call__(self, structure):
        structure['antigen'] = None
        structure['antigen_seqmap'] = None
        return structure


class SelectAtom:
    """
    Selects atom coordinates from the structure based on the resolution (full or backbone).

    Parameters
    ----------
    resolution : str
        The resolution for selecting atoms ('full' or 'backbone').
    """
    def __init__(self, resolution):
        assert resolution in ('full', 'backbone')
        self.resolution = resolution

    def __call__(self, data):
        """
        Updates the structure data by selecting atom coordinates based on resolution.

        Parameters
        ----------
        data : dict
            Structure data containing atom positions and masks.

        Returns
        -------
        dict
            Updated structure data with selected atom positions and masks.
        """
        data['pos_atoms'] = data['pos_heavyatom'] if self.resolution == 'full' else data['pos_heavyatom'][:, :5]
        data['mask_atoms'] = data['mask_heavyatom'] if self.resolution == 'full' else data['mask_heavyatom'][:, :5]
        return data


class RemoveNative:
    """
    Removes native sequence or structure information based on configuration settings.

    Parameters
    ----------
    remove_structure : bool
        Whether to remove the native structure.
    remove_sequence : bool
        Whether to remove the native sequence.
    """
    def __init__(self, remove_structure, remove_sequence):
        self.remove_structure = remove_structure
        self.remove_sequence = remove_sequence

    def __call__(self, data):
        """
        Removes sequence or structure information for regions marked by the generation mask.

        Parameters
        ----------
        data : dict
            Structure data containing sequences and positions.

        Returns
        -------
        dict
            Updated structure data with removed sequences or structures.
        """
        generation_mask = data['generation_mask'].clone()

        # Remove sequence information
        if self.remove_sequence:
            data['aa'] = torch.where(
                generation_mask, 
                torch.full_like(data['aa'], fill_value=int(AALib.UNK)),
                data['aa']
            )

        # Remove structure information
        if self.remove_structure:
            data['pos_heavyatom'] = torch.where(
                generation_mask[:, None, None].expand(data['pos_heavyatom'].shape),
                torch.randn_like(data['pos_heavyatom']) * 10,
                data['pos_heavyatom']
            )

        return data
