"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: Library of models and related support functions for protein structure and sequence prediction.
"""

import copy
import functools
import math
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm.auto import tqdm
from torch import einsum, nn

from lightning import Callback
from lightning.pytorch.callbacks import BasePredictionWriter

from utils.geometry import (apply_rotation_to_vector, construct_3d_basis, get_3d_basis, get_bb_dihedral_angles,
                            global2local, local2global, normalize_vector, pairwise_dihedrals, quaternion_1ijk_to_rotation_matrix,
                            random_uniform_so3, randn_so3, rotation_to_so3vec, so3_vec2rotation)
from utils.loss_functions import rotation_loss
from utils.protein_constants import AALib, BBHeavyAtom, CDR, resindex_to_ressymb, restype_to_heavyatom_names
from utils.utils_diff import clampped_one_hot

class DiffAbXL(nn.Module):
    """
    DiffAbXL model for protein sequence and structure denoising and generation.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model hyperparameters and settings.

    Attributes
    ----------
    config : dict
        Model configuration settings.
    train_structure : bool
        Flag to indicate whether to train structure denoising.
    train_sequence : bool
        Flag to indicate whether to train sequence denoising.
    residue_emb : ResidueEmbedding
        Embedding module for residue-level embeddings.
    pair_emb : PairEmbedding
        Embedding module for pairwise residue embeddings.
    diffusion : Diffusion
        Diffusion model for sequence and structure generation.
    """
    def __init__(self, config):
        super().__init__()

        # Device configuration and model settings
        config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.train_structure = self.config['train_structure']
        self.train_sequence = self.config['train_sequence']

        # Embedding dimensions and model components
        num_atoms = config['num_atoms']
        residue_dim = config['residue_dim']
        pair_dim = config['pair_dim']

        # Residue and pair embeddings
        self.residue_emb = ResidueEmbedding(residue_dim, num_atoms, max_aa_types=config['max_aa_types'], max_chain_types=10)
        self.pair_emb = PairEmbedding(pair_dim, num_atoms, max_aa_types=config['max_aa_types'], max_relpos=32)

        # Diffusion model
        self.diffusion = Diffusion(self.config)

    def forward(self, batch):
        """
        Forward pass through the model.

        Parameters
        ----------
        batch : dict
            Input batch containing sequence, structure, and mask information.

        Returns
        -------
        loss_dict : dict
            Dictionary of loss values for the current batch.
        """
        # Extract generation and residue masks
        generation_mask = batch['generation_mask']
        residue_mask = batch['residue_mask']

        # Encode batch to extract initial embeddings and positions
        v0, p0, s0, res_emb, pair_emb = self.encode_batch(batch)

        # Compute diffusion-based loss and predictions
        loss_dict, eps_p_pred, Rpred, R0, res_emb_intermediate = self.diffusion(
            v0, p0, s0, res_emb, pair_emb, generation_mask, residue_mask, 
            denoise_structure=self.train_structure, denoise_sequence=self.train_sequence
        )

        return loss_dict

    def encode_batch(self, batch):
        """
        Encode the input batch to get residue embeddings, pair embeddings, 
        initial AA sequence, and position of atoms.

        Parameters
        ----------
        batch : dict
            Input batch containing residue sequence and structural information.

        Returns
        -------
        v0 : torch.Tensor
            SO(3) vector representation of rotations for each residue.
        p0 : torch.Tensor
            Positions of C-alpha atoms.
        s0 : torch.Tensor
            Initial amino acid sequence.
        res_emb : torch.Tensor
            Residue-level embeddings.
        pair_emb : torch.Tensor
            Pairwise residue embeddings.
        """
        # Extract sequence, fragment type, and heavy atom positional information
        s0 = batch['aa']
        res_nb = batch['res_nb']
        fragment_type = batch['fragment_type']
        pos_heavyatom = batch['pos_heavyatom']
        mask_heavyatom = batch['mask_heavyatom']
        generation_mask_bar = ~batch['generation_mask']

        # Construct context masks for training structure and sequence
        context_mask = torch.logical_and(
            batch['mask_heavyatom'][:, :, BBHeavyAtom.CA], 
            ~batch['generation_mask']
        )
        structure_mask = context_mask if self.train_structure else None
        sequence_mask = context_mask if self.train_sequence else None

        # Compute residue embeddings
        res_emb = self.residue_emb(
            aa=s0, res_nb=res_nb, fragment_type=fragment_type, 
            pos_atoms=pos_heavyatom, mask_atoms=mask_heavyatom, 
            structure_mask=structure_mask, sequence_mask=sequence_mask, 
            generation_mask_bar=generation_mask_bar
        )

        # Compute pairwise residue embeddings
        pair_emb = self.pair_emb(
            aa=s0, res_nb=res_nb, fragment_type=fragment_type, 
            pos_atoms=pos_heavyatom, mask_atoms=mask_heavyatom, 
            structure_mask=structure_mask, sequence_mask=sequence_mask
        )

        # Extract positions of C-alpha atoms and construct 3D basis
        p0 = pos_heavyatom[:, :, BBHeavyAtom.CA]
        R0 = construct_3d_basis(
            center=pos_heavyatom[:, :, BBHeavyAtom.CA], 
            p1=pos_heavyatom[:, :, BBHeavyAtom.C], 
            p2=pos_heavyatom[:, :, BBHeavyAtom.N]
        )
        v0 = rotation_to_so3vec(R0)

        return v0, p0, s0, res_emb, pair_emb

    @torch.no_grad()
    def sample(self, batch, sample_structure=True, sample_sequence=True):
        """
        Sample new sequences and structures using the diffusion model.

        Parameters
        ----------
        batch : dict
            Input batch containing initial positions and sequences.
        sample_structure : bool, optional
            Flag indicating whether to sample structure, by default True.
        sample_sequence : bool, optional
            Flag indicating whether to sample sequence, by default True.

        Returns
        -------
        traj : torch.Tensor
            Trajectory of sampled positions and sequences.
        """
        generation_mask = batch['generation_mask']
        residue_mask = batch['residue_mask']

        # Encode the batch
        v0, p0, s0, res_emb, pair_emb = self.encode_batch(batch)

        # Sample trajectory
        traj, _ = self.diffusion.sample(
            v0, p0, s0, res_emb, pair_emb, generation_mask, residue_mask, 
            sample_structure=sample_structure, sample_sequence=sample_sequence
        )
        return traj

    @torch.no_grad()
    def optimize(self, batch, opt_step, sample_structure=True, sample_sequence=True):
        """
        Perform optimization step using the diffusion model.

        Parameters
        ----------
        batch : dict
            Input batch containing initial positions and sequences.
        opt_step : int
            Optimization step number.
        sample_structure : bool, optional
            Flag indicating whether to optimize structure, by default True.
        sample_sequence : bool, optional
            Flag indicating whether to optimize sequence, by default True.

        Returns
        -------
        traj : torch.Tensor
            Trajectory of positions and sequences.
        """
        generation_mask = batch['generation_mask']
        residue_mask = batch['residue_mask']

        # Encode the batch
        v0, p0, s0, res_emb, pair_emb = self.encode_batch(batch)

        # Perform optimization and return trajectory
        traj = self.diffusion.optimize(
            opt_step, v0, p0, s0, res_emb, pair_emb, generation_mask, residue_mask, 
            sample_structure=sample_structure, sample_sequence=sample_sequence
        )
        return traj

    @torch.no_grad()
    def get_posterior(self, batch, sample_structure=False, sample_sequence=False):
        """
        Compute posterior of the sampled sequences.

        Parameters
        ----------
        batch : dict
            Input batch containing sequence and structure information.
        sample_structure : bool, optional
            Flag indicating whether to sample structure, by default False.
        sample_sequence : bool, optional
            Flag indicating whether to sample sequence, by default False.

        Returns
        -------
        seq_org : torch.Tensor
            Original sequences from the batch.
        subpos : torch.Tensor
            Subpositions predicted by the model.
        """
        generation_mask = batch['generation_mask']
        residue_mask = batch['residue_mask']

        # Encode the batch
        v0, p0, s0, res_emb, pair_emb = self.encode_batch(batch)

        # Sample from the diffusion model
        traj, post = self.diffusion.sample(
            v0, p0, s0, res_emb, pair_emb, generation_mask, residue_mask, 
            sample_structure=sample_structure, sample_sequence=sample_sequence, 
            move_to_cpu=False
        )

        # Extract the original sequence and predicted subpositions
        subpos = post[batch['generation_mask']].view(p0.size(0), -1, 20)
        seq_org = batch['original_seq'][batch['generation_mask']].view(p0.size(0), -1)

        return seq_org, subpos


class Diffusion(nn.Module):
    """
    Diffusion model for sequence and structure denoising and generation.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model hyperparameters and settings.

    Attributes
    ----------
    config : dict
        Model configuration settings.
    num_steps : int
        Number of diffusion steps.
    eps_net : EpsilonNet
        Network to predict noise in the denoising process.
    rot_trans : RotationTransition
        Transition model for rotations.
    pos_trans : PositionTransition
        Transition model for positions.
    seq_trans : SequenceTransition
        Transition model for sequences.
    position_mean : torch.Tensor
        Mean used to normalize positions.
    position_scale : torch.Tensor
        Scale used to normalize positions.
    """
    def __init__(self, config):
        super(Diffusion, self).__init__()

        self.config = config
        self.num_steps = config["num_steps"]
        
        # Define noise predictor
        self.eps_net = EpsilonNet(config)
        
        # Transitions for rotation, position, and sequence
        self.rot_trans = RotationTransition(config)
        self.pos_trans = PositionTransition(config)
        self.seq_trans = SequenceTransition(config) if config["noising_scheme"] == "uniform" else SequenceTransitionMasked(config)
        
        # Register buffers for position normalization
        self.register_buffer('position_mean', torch.FloatTensor(config['position_mean']).view(1, 1, -1))
        self.register_buffer('position_scale', torch.FloatTensor(config['position_scale']).view(1, 1, -1))
        self.register_buffer('_dummy', torch.empty([0, ]))
        
    def forward(self, v0, p0, s0, res_emb, pair_emb, generation_mask, residue_mask, denoise_structure, denoise_sequence, t=None):
        """
        Forward pass through the diffusion process.

        Parameters
        ----------
        v0 : torch.Tensor
            Initial rotations (SO(3) vectors).
        p0 : torch.Tensor
            Initial positions (C-alpha atom positions).
        s0 : torch.Tensor
            Initial sequence (amino acid indices).
        res_emb : torch.Tensor
            Residue-level embeddings.
        pair_emb : torch.Tensor
            Pairwise residue embeddings.
        generation_mask : torch.Tensor
            Mask for indicating which residues are generated.
        residue_mask : torch.Tensor
            Mask for valid residues.
        denoise_structure : bool
            Flag to indicate whether to denoise structure (rotation, position).
        denoise_sequence : bool
            Flag to indicate whether to denoise sequence.
        t : torch.Tensor, optional
            Time steps for the diffusion process, by default None.

        Returns
        -------
        loss_dict : dict
            Dictionary of computed loss values.
        eps_p_pred : torch.Tensor
            Predicted noise for positions.
        Rpred : torch.Tensor
            Predicted rotation matrices.
        R0 : torch.Tensor
            Ground truth rotation matrices.
        res_emb_intermediate : torch.Tensor
            Intermediate residue embeddings during the denoising process.
        """
        N, L = res_emb.shape[:2]
        loss_dict = {}

        # 1--- Prepare time steps and normalize position
        if t is None:
            t = torch.randint(0, self.num_steps, (N,), dtype=torch.long, device=p0.device)

        # Get beta values and normalize the initial position
        beta = self.pos_trans.var_schedule.betas[t]
        p0 = self._normalize_position(p0)

        # Get rotation matrices from SO(3) vectors
        R0 = so3_vec2rotation(v0)
        
        # 2--- Apply diffusion (add noise)
        if denoise_structure:
            # Add noise to rotation and position
            v_noisy, _ = self.rot_trans.add_noise(v0, generation_mask, t)
            p_noisy, eps_p = self.pos_trans.add_noise(p0, generation_mask, t)
        else:
            v_noisy = v0.clone()
            p_noisy = p0.clone()

        if denoise_sequence:
            # Add noise to sequence
            _, s_noisy = self.seq_trans.add_noise(s0, generation_mask, t)
        else:
            s_noisy = s0.clone()
            
        # 3--- Predict denoised values
        v_pred, Rpred, eps_p_pred, c_denoised, residue_time_emb, res_emb_intermediate = self.eps_net(
            v_noisy, p_noisy, s_noisy, res_emb, pair_emb, beta, generation_mask, residue_mask
        )

        # 4--- Compute losses
        # Rotation loss
        loss_rot = rotation_loss(Rpred, R0)
        loss_dict = self.mask_out_loss(loss_rot, generation_mask, loss_dict, loss_type='rot')

        # Position loss
        loss_pos = F.mse_loss(eps_p_pred, eps_p, reduction='none').sum(dim=-1)
        loss_dict = self.mask_out_loss(loss_pos, generation_mask, loss_dict, loss_type='pos')

        # Sequence loss (KL divergence)
        posterior_true = self.seq_trans.posterior(s_noisy, s0, t)
        log_posterior_pred = torch.log(self.seq_trans.posterior(s_noisy, c_denoised, t) + 1e-8)
        kl_div = F.kl_div(log_posterior_pred, posterior_true, reduction='none').sum(dim=-1)
        loss_dict = self.mask_out_loss(kl_div, generation_mask, loss_dict, loss_type='seq')

        return loss_dict, eps_p_pred, Rpred, R0, res_emb_intermediate

    @torch.no_grad() 
    def sample(self, v, p, s, res_emb, pair_emb, generation_mask, residue_mask, sample_structure=True, sample_sequence=True, pbar=False, move_to_cpu=True):
        """
        Sample new sequences and structures using reverse diffusion.

        Parameters
        ----------
        v : torch.Tensor
            Initial orientations of residues.
        p : torch.Tensor
            Initial positions of residues.
        s : torch.Tensor
            Initial sequence of residues.
        res_emb : torch.Tensor
            Residue-level embeddings.
        pair_emb : torch.Tensor
            Pairwise residue embeddings.
        generation_mask : torch.Tensor
            Mask for generated residues.
        residue_mask : torch.Tensor
            Mask for valid residues.
        sample_structure : bool, optional
            Flag to indicate whether to sample structure, by default True.
        sample_sequence : bool, optional
            Flag to indicate whether to sample sequence, by default True.
        pbar : bool, optional
            Flag to indicate whether to show a progress bar, by default False.
        move_to_cpu : bool, optional
            Flag to indicate whether to move intermediate results to CPU, by default True.

        Returns
        -------
        traj : dict
            Trajectory of sampled rotations, positions, and sequences across time steps.
        s_post : torch.Tensor
            Posterior sequence probabilities after sampling.
        """
        N, L = res_emb.shape[:2]

        # Normalize initial position
        p = self._normalize_position(p)

        # 1--- Initialize random values for structure and sequence
        v_init, p_init, s_init = v, p, s
        if sample_structure:
            v_rand = random_uniform_so3([N, L], device=v.device)
            p_rand = torch.randn_like(p)
            v_init = torch.where(generation_mask[:, :, None].expand_as(v), v_rand, v)
            p_init = torch.where(generation_mask[:, :, None].expand_as(p), p_rand, p)

        if sample_sequence:
            s_rand = torch.randint_like(s, low=0, high=19)
            s_init = torch.where(generation_mask, s_rand, s)        

        # 2--- Run reverse diffusion to sample structure and sequence
        traj, s_post = self.run_reverse_diffusion(
            v_init, p_init, s_init, res_emb, pair_emb, generation_mask, residue_mask, time_step=self.num_steps, desc='Sampling...', move_to_cpu=move_to_cpu
        )
        
        return traj, s_post

    @torch.no_grad()
    def optimize(self, time_step, v, p, s, res_emb, pair_emb, generation_mask, residue_mask, sample_structure=True, sample_sequence=True, pbar=False, move_to_cpu=False):
        """
        Optimize the denoising process by adding noise and then denoising.

        Parameters
        ----------
        time_step : int
            The current time step in the diffusion process.
        v : torch.Tensor
            Initial orientations of residues.
        p : torch.Tensor
            Initial positions of residues.
        s : torch.Tensor
            Initial sequence of residues.
        res_emb : torch.Tensor
            Residue-level embeddings.
        pair_emb : torch.Tensor
            Pairwise residue embeddings.
        generation_mask : torch.Tensor
            Mask for generated residues.
        residue_mask : torch.Tensor
            Mask for valid residues.
        sample_structure : bool, optional
            Flag to indicate whether to optimize structure, by default True.
        sample_sequence : bool, optional
            Flag to indicate whether to optimize sequence, by default True.
        pbar : bool, optional
            Flag to indicate whether to show a progress bar, by default False.
        move_to_cpu : bool, optional
            Flag to indicate whether to move intermediate results to CPU, by default False.

        Returns
        -------
        traj : dict
            Optimized trajectory of rotations, positions, and sequences.
        s_post : torch.Tensor
            Posterior sequence probabilities after optimization.
        """
        N, L = res_emb.shape[:2]

        # Normalize initial position
        p = self._normalize_position(p)
        t = torch.full([N, ], fill_value=time_step, dtype=torch.long, device=p.device)
        
        # 1--- Add noise to structure and sequence
        v_init, p_init, s_init = v, p, s
        if sample_structure:
            v_noisy, _ = self.rot_trans.add_noise(v, generation_mask, t)
            p_noisy, _ = self.pos_trans.add_noise(p, generation_mask, t)
            v_init = torch.where(generation_mask[:, :, None].expand_as(v), v_noisy, v)
            p_init = torch.where(generation_mask[:, :, None].expand_as(p), p_noisy, p)

        if sample_sequence:
            _, s_noisy = self.seq_trans.add_noise(s, generation_mask, t)
            s_init = torch.where(generation_mask, s_noisy, s)
            
        # 2--- Run reverse diffusion for optimization
        traj, s_post = self.run_reverse_diffusion(
            v_init, p_init, s_init, res_emb, pair_emb, generation_mask, residue_mask, time_step=time_step, desc='Optimizing...', move_to_cpu=move_to_cpu
        )
        
        return traj, s_post

    @torch.no_grad() 
    def run_reverse_diffusion(self, v_init, p_init, s_init, res_emb, pair_emb, generation_mask, residue_mask, time_step=100, desc='Sampling...', move_to_cpu=True):
        """
        Run the reverse diffusion process to denoise and sample structure and sequence.

        Parameters
        ----------
        v_init : torch.Tensor
            Initial orientations of residues.
        p_init : torch.Tensor
            Initial positions of residues.
        s_init : torch.Tensor
            Initial sequence of residues.
        res_emb : torch.Tensor
            Residue-level embeddings.
        pair_emb : torch.Tensor
            Pairwise residue embeddings.
        generation_mask : torch.Tensor
            Mask for generated residues.
        residue_mask : torch.Tensor
            Mask for valid residues.
        time_step : int, optional
            Total number of time steps, by default 100.
        desc : str, optional
            Description for progress bar, by default 'Sampling...'.
        move_to_cpu : bool, optional
            Flag to indicate whether to move intermediate results to CPU, by default True.

        Returns
        -------
        traj : dict
            Trajectory of rotations, positions, and sequences at each time step.
        s_post : torch.Tensor
            Posterior sequence probabilities after reverse diffusion.
        """
        N, L = v_init.shape[:2]
        traj = {time_step: (v_init, self._unnormalize_position(p_init), s_init)}
        
        pbar = functools.partial(tqdm, total=time_step, desc=desc) if self.pbar else lambda x: x
            
        for t in pbar(range(time_step, 0, -1)):
            # Get values at current time step
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)
            beta = self.pos_trans.var_schedule.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=p_t.device)

            # Predict denoised values
            v_tm1, R_tm1, eps_p, c_denoised, residue_time_emb, _ = self.eps_net(
                v_t, p_t, s_t, res_emb, pair_emb, beta, generation_mask, residue_mask
            )

            # Denoise rotation, position, and sequence
            v_tm1 = self.rot_trans.denoise(v_t, v_tm1, generation_mask, t_tensor)
            p_tm1 = self.pos_trans.denoise(p_t, eps_p, generation_mask, t_tensor)
            s_post, s_tm1 = self.seq_trans.denoise(s_t, c_denoised, generation_mask, t_tensor)

            if not self.sample_structure:
                v_tm1, p_tm1 = v_t, p_t
            if not self.sample_sequence:
                s_tm1 = s_t

            # Record the trajectory at (t-1)
            traj[t-1] = (v_tm1, self._unnormalize_position(p_tm1), s_tm1)

            # Optionally move to CPU to save memory
            if move_to_cpu:
                traj[t] = tuple(x.cpu() for x in traj[t])

        return traj, s_post

    def mask_out_loss(self, loss, mask, loss_dict, loss_type='rot'):
        """
        Apply mask to loss and add it to the loss dictionary.

        Parameters
        ----------
        loss : torch.Tensor
            Loss tensor to be masked.
        mask : torch.Tensor
            Mask to apply to the loss.
        loss_dict : dict
            Dictionary to store the masked loss.
        loss_type : str, optional
            Type of loss ('rot', 'pos', 'seq'), by default 'rot'.

        Returns
        -------
        dict
            Updated loss dictionary.
        """
        loss_dict[loss_type] = (loss * mask).sum() / (mask.sum().float() + 1e-8)
        return loss_dict
        
    def _normalize_position(self, p):
        """
        Normalize positions by subtracting the mean and scaling by standard deviation.

        Parameters
        ----------
        p : torch.Tensor
            Input positions to normalize.

        Returns
        -------
        torch.Tensor
            Normalized positions.
        """
        return (p - self.position_mean) / self.position_scale
        
    def _unnormalize_position(self, p):
        """
        Unnormalize positions by adding the mean and scaling by standard deviation.

        Parameters
        ----------
        p : torch.Tensor
            Input normalized positions to unnormalize.

        Returns
        -------
        torch.Tensor
            Unnormalized positions.
        """
        return p * self.position_scale + self.position_mean


class EpsilonNet(nn.Module):
    """
    Epsilon network for predicting noise during denoising in the diffusion process.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model hyperparameters.

    Attributes
    ----------
    config : dict
        Model configuration settings.
    residue_dim : int
        Dimensionality of residue embeddings.
    pair_dim : int
        Dimensionality of pairwise embeddings.
    num_layers : int
        Number of layers in the network.
    seq_emb : nn.Embedding
        Embedding layer for sequences (amino acids).
    residue_encoder : nn.Sequential
        MLP for encoding residue embeddings.
    att_encoder : ResPairformer
        Attention-based encoder for residues and pair embeddings.
    eps_pos_net : nn.Sequential
        Network for predicting noise in position.
    eps_rot_net : nn.Sequential
        Network for predicting noise in rotation.
    eps_seq_net : nn.Sequential
        Network for predicting sequence.
    """
    def __init__(self, config):
        super(EpsilonNet, self).__init__()
        
        self.config = config
        self.residue_dim = config['residue_dim']
        self.pair_dim = config['pair_dim']  
        self.num_layers = config['num_layers']
        
        # Sequence embedding for amino acids (25 unique tokens)
        self.seq_emb = nn.Embedding(25, self.residue_dim)
        
        # Residue encoder (MLP for encoding combined residue embeddings)
        self.residue_encoder = nn.Sequential(
            nn.Linear(2 * self.residue_dim, self.residue_dim), 
            nn.ReLU(), 
            nn.Linear(self.residue_dim, self.residue_dim),
        )

        # Attention-based encoder for residue and pair embeddings
        self.att_encoder = ResPairformer(config)
        
        # Networks for predicting position, rotation, and sequence noise
        self.eps_pos_net = self.get_eps_network(sequence=False, out_dim=3)
        self.eps_rot_net = self.get_eps_network(sequence=False, out_dim=3)
        self.eps_seq_net = self.get_eps_network(sequence=True, out_dim=20)

    def get_eps_network(self, sequence=False, out_dim=3):
        """
        Create the epsilon network for predicting noise in either structure or sequence.

        Parameters
        ----------
        sequence : bool, optional
            If True, create network for sequence prediction, by default False.
        out_dim : int, optional
            Output dimensionality of the network, by default 3.

        Returns
        -------
        nn.Sequential
            A sequential network for noise prediction.
        """
        # Network architecture for noise prediction
        modules = [
            nn.Linear(self.residue_dim + 3, self.residue_dim), 
            nn.ReLU(),
            nn.Linear(self.residue_dim, self.residue_dim), 
            nn.ReLU(),
            nn.Linear(self.residue_dim, out_dim)
        ]
        
        if sequence:
            # For sequence prediction, apply softmax at the output layer
            modules.append(nn.Softmax(dim=-1))
        
        return nn.Sequential(*modules)

    def forward(self, v_t, p_t, s_t, res_emb, pair_emb, beta, generation_mask, residue_mask):
        """
        Forward pass through the Epsilon network for predicting position, rotation, and sequence.

        Parameters
        ----------
        v_t : torch.Tensor
            Current orientations of residues (SO(3) vectors), shape (N, L, 3).
        p_t : torch.Tensor
            Current positions of residues (C-alpha positions), shape (N, L, 3).
        s_t : torch.Tensor
            Current sequence of residues (amino acid indices), shape (N, L).
        res_emb : torch.Tensor
            Residue-level embeddings, shape (N, L, residue_dim).
        pair_emb : torch.Tensor
            Pairwise residue embeddings, shape (N, L, L, pair_dim).
        beta : torch.Tensor
            Time-dependent noise scaling factor, shape (N,).
        generation_mask : torch.Tensor
            Mask indicating which residues are generated, shape (N, L).
        residue_mask : torch.Tensor
            Mask indicating valid residues, shape (N, L).

        Returns
        -------
        v_tm1 : torch.Tensor
            Predicted orientation (SO(3) vectors) at t-1, shape (N, L, 3).
        R_tm1 : torch.Tensor
            Predicted rotation matrix at t-1, shape (N, L, 3, 3).
        eps_pos : torch.Tensor
            Predicted noise in position, shape (N, L, 3).
        c_denoised : torch.Tensor
            Denoised categorical distribution over sequences, shape (N, L, 20).
        residue_time_emb : torch.Tensor
            Combined residue and time embeddings used for prediction, shape (N, L, residue_dim+3).
        res_emb : torch.Tensor
            Updated residue embeddings after attention encoding, shape (N, L, residue_dim).
        """
        N, L = v_t.shape[:2]

        # Get rotation matrices from SO(3) vectors
        R_t = so3_vec2rotation(v_t)
        
        # 1--- Embed residues and update embeddings
        # Concatenate the initial and current sequence embeddings, and encode them
        res_emb_t = self.seq_emb(s_t)  # Embed current sequence
        res_emb_cat = torch.cat([res_emb, res_emb_t], dim=-1)  # Concatenate with previous embeddings
        res_emb = self.residue_encoder(res_emb_cat)  # Encode concatenated embeddings
        
        # Apply attention encoder to the updated residue and pair embeddings
        res_emb = self.att_encoder(R_t, p_t, res_emb, pair_emb, residue_mask)
        
        # 2--- Combine updated residue embedding with time embedding
        # Create time embedding from beta and concatenate with residue embeddings
        t_embed = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)[:, None, :].expand(N, L, 3)
        residue_time_emb = torch.cat([res_emb, t_embed], dim=-1)
        
        # 3--- Predict position noise
        eps_pos1 = self.eps_pos_net(residue_time_emb)  # Predict noise for position
        eps_pos2 = apply_rotation_to_vector(R_t, eps_pos1)  # Apply rotation to the predicted noise
        eps_pos3 = torch.where(generation_mask[:, :, None].expand_as(eps_pos2), eps_pos2, torch.zeros_like(eps_pos2))
        
        # 4--- Predict rotation noise
        eps_rot = self.eps_rot_net(residue_time_emb)  # Predict noise for rotation
        U = quaternion_1ijk_to_rotation_matrix(eps_rot)  # Convert quaternion to rotation matrix
        R_tm1 = R_t @ U  # Apply rotation update
        v_tm1 = rotation_to_so3vec(R_tm1)  # Convert back to SO(3) vectors
        v_tm1 = torch.where(generation_mask[:, :, None].expand_as(v_tm1), v_tm1, v_t)  # Mask out non-generated regions

        # 5--- Predict sequence noise (already softmaxed)
        c_denoised = self.eps_seq_net(residue_time_emb)

        return v_tm1, R_tm1, eps_pos3, c_denoised, residue_time_emb, res_emb

class ResPairformer(nn.Module):
    """
    Residue-Pairformer network consisting of multiple ResPairBlock layers,
    which perform attention-based interactions between residue and pair embeddings.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model hyperparameters.

    Attributes
    ----------
    blocks : nn.ModuleList
        A list of ResPairBlock layers used for hierarchical attention-based feature encoding.
    """
    def __init__(self, config):
        super(ResPairformer, self).__init__()
        
        residue_dim = config['residue_dim']
        pair_dim = config['pair_dim']
        num_layers = config['num_layers']
        
        # List of Residue-Pair attention blocks
        self.blocks = nn.ModuleList([ResPairBlock(residue_dim, pair_dim) for _ in range(num_layers)])
        
    def forward(self, R, t, res_feat, pair_feat, mask):
        """
        Forward pass through the ResPairformer.

        Parameters
        ----------
        R : torch.Tensor
            Frame basis matrices, shape (N, L, 3, 3).
        t : torch.Tensor
            Frame external (absolute) coordinates, shape (N, L, 3).
        res_feat : torch.Tensor
            Node-wise features (residue features), shape (N, L, F).
        pair_feat : torch.Tensor
            Pair-wise features (residue pair interactions), shape (N, L, L, C).
        mask : torch.Tensor
            Masks for valid residues, shape (N, L).

        Returns
        -------
        torch.Tensor
            Updated node-wise features (residue features), shape (N, L, F).
        """
        for block in self.blocks:
            res_feat = block(R, t, res_feat, pair_feat, mask)
        return res_feat
    

class ResPairBlock(nn.Module):
    """
    Residue-Pair attention block that performs node-wise, pair-wise, and spatial attention
    between residue features using learned attention weights.

    Parameters
    ----------
    residue_dim : int
        Dimensionality of residue features.
    pair_dim : int
        Dimensionality of pair-wise features.
    q_dim : int, optional
        Dimensionality of query projections, by default 32.
    v_dim : int, optional
        Dimensionality of value projections, by default 32.
    num_q_points : int, optional
        Number of query points, by default 8.
    num_v_points : int, optional
        Number of value points, by default 8.
    num_heads : int, optional
        Number of attention heads, by default 12.
    bias : bool, optional
        Whether to include bias in linear projections, by default False.
    """
    def __init__(self, residue_dim, pair_dim, q_dim=32, v_dim=32, num_q_points=8, num_v_points=8, num_heads=12, bias=False):
        super(ResPairBlock, self).__init__()
        
        self.residue_dim = residue_dim
        self.pair_dim = pair_dim
        self.q_dim = q_dim
        self.v_dim = v_dim
        self.num_q_points = num_q_points
        self.num_v_points = num_v_points
        self.num_heads = num_heads
        self.bias = bias
        
        # Node layers (query, key, value projections)
        self.q_proj = nn.Linear(residue_dim, q_dim * num_heads, bias=bias)
        self.k_proj = nn.Linear(residue_dim, q_dim * num_heads, bias=bias)
        self.v_proj = nn.Linear(residue_dim, v_dim * num_heads, bias=bias)
        
        # Pair layer (pair-wise attention bias projection)
        self.pair_bias_proj = nn.Linear(pair_dim, num_heads, bias=bias)
        
        # Spatial coefficient (log-scaling for softplus)
        coeff = torch.full([1, 1, 1, num_heads], fill_value=np.log(np.exp(1.0) - 1.0))
        self.spatial_coef = nn.Parameter(coeff, requires_grad=True)
        
        # Projections for spatial query, key, and value
        self.q_point_proj = nn.Linear(residue_dim, num_q_points * num_heads * 3, bias=bias)
        self.k_point_proj = nn.Linear(residue_dim, num_q_points * num_heads * 3, bias=bias)
        self.v_point_proj = nn.Linear(residue_dim, num_v_points * num_heads * 3, bias=bias)

        # Output transformation layer
        input_features = (num_heads * v_dim) + (num_heads * pair_dim) + (num_heads * num_v_points * (3+3+1))
        self.out_transform = nn.Linear(input_features, residue_dim)
        
        # MLP transition for residual updates
        self.mlp_transition = nn.Sequential(
            nn.Linear(residue_dim, residue_dim), 
            nn.ReLU(), 
            nn.Linear(residue_dim, residue_dim), 
            nn.ReLU(), 
            nn.Linear(residue_dim, residue_dim)
        )
        
        # Layer normalization
        self.layer_norm_1 = nn.LayerNorm(residue_dim)
        self.layer_norm_2 = nn.LayerNorm(residue_dim)

    def forward(self, R, coord, residue_feat, pair_feat, mask):
        """
        Forward pass through the Residue-Pair block.

        Parameters
        ----------
        R : torch.Tensor
            Frame basis matrices, shape (N, L, 3, 3).
        coord : torch.Tensor
            Frame external (absolute) coordinates, shape (N, L, 3).
        residue_feat : torch.Tensor
            Node-wise features (residue features), shape (N, L, F).
        pair_feat : torch.Tensor
            Pair-wise features, shape (N, L, L, C).
        mask : torch.Tensor
            Mask indicating valid residues, shape (N, L).

        Returns
        -------
        torch.Tensor
            Updated node-wise features, shape (N, L, F).
        """
        # 1--- Compute attention logits for node, pair, and spatial components
        logits_node = self._node_logits(residue_feat)
        logits_pair = self._pair_logits(pair_feat)
        logits_spatial = self._spatial_logits(R, coord, residue_feat)
        
        # Combine logits and compute attention weights (alpha)
        logits_sum = (logits_node + logits_pair + logits_spatial) * np.sqrt(1/3)
        alpha = self._alpha_from_logits(logits_sum, mask)
        
        # 2--- Aggregate features from nodes, pairs, and spatial information
        feat_node = self._node_aggregation(alpha, residue_feat)
        feat_pair = self._pair_aggregation(alpha, pair_feat)
        feat_spatial = self._spatial_aggregation(alpha, R, coord, residue_feat)
        
        # 3--- Update node embeddings with the aggregated features
        feat_all = torch.cat([feat_pair, feat_node, feat_spatial], dim=-1)
        feat_all = self.out_transform(feat_all)
        feat_all = self.mask_zero(mask.unsqueeze(-1), feat_all)
        residue_feat_updated = self.layer_norm_1(residue_feat + feat_all)
        residue_feat_updated = self.layer_norm_2(residue_feat_updated + self.mlp_transition(residue_feat_updated))
        
        return residue_feat_updated

    def mask_zero(self, mask, value):
        """
        Apply masking to avoid invalid updates for padded residues.

        Parameters
        ----------
        mask : torch.Tensor
            Mask for valid residues, shape (N, L, 1).
        value : torch.Tensor
            Values to be masked, shape (N, L, F).

        Returns
        -------
        torch.Tensor
            Masked values, shape (N, L, F).
        """
        return torch.where(mask, value, torch.zeros_like(value))


    def _alpha_from_logits(self, logits, mask, inf=1e5):
        """
        Compute attention weights (alpha) from logits.

        Parameters
        ----------
        logits : torch.Tensor
            Logit matrices, shape (N, L, L, num_heads).
        mask : torch.Tensor
            Mask for valid residues, shape (N, L).
        inf : float, optional
            Large negative value to apply for masking, by default 1e5.

        Returns
        -------
        torch.Tensor
            Attention weights (alpha), shape (N, L, L, num_heads).
        """
        N, L, _, _ = logits.size()
        mask_row = mask.view(N, L, 1, 1).expand_as(logits)
        mask_pair = mask_row * mask_row.permute(0, 2, 1, 3)

        logits = torch.where(mask_pair, logits, logits - inf)
        alpha = torch.softmax(logits, dim=2)
        alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
        return alpha
        
    def _node_logits(self, residue_feat):
        """
        Compute node-wise attention logits from residue features.

        Parameters
        ----------
        residue_feat : torch.Tensor
            Residue features, shape (N, L, residue_dim).

        Returns
        -------
        torch.Tensor
            Node-wise attention logits, shape (N, L, L, num_heads).
        """
        N, L, _ = residue_feat.size()

        # Project residue features to query and key vectors
        q = self.q_proj(residue_feat).view(N, L, self.num_heads, self.q_dim)
        k = self.k_proj(residue_feat).view(N, L, self.num_heads, self.q_dim)
        
        # Compute attention logits
        logits = q.unsqueeze(2) * k.unsqueeze(1) * (1 / np.sqrt(self.q_dim))
        logits = logits.sum(-1)
        
        return logits

    def _pair_logits(self, pair_feat):
        """
        Compute pair-wise attention logits from pair features.

        Parameters
        ----------
        pair_feat : torch.Tensor
            Pair-wise features, shape (N, L, L, pair_dim).

        Returns
        -------
        torch.Tensor
            Pair-wise attention logits, shape (N, L, L, num_heads).
        """
        return self.pair_bias_proj(pair_feat)

    def _compute_q_or_k(self, residue_feat, R, coord, fn):
        """
        Compute query or key vectors in global frame of reference.

        Parameters
        ----------
        residue_feat : torch.Tensor
            Residue features, shape (N, L, residue_dim).
        R : torch.Tensor
            Frame basis matrices, shape (N, L, 3, 3).
        coord : torch.Tensor
            Frame external coordinates, shape (N, L, 3).
        fn : callable
            Function for projection (for query or key).

        Returns
        -------
        torch.Tensor
            Projected query or key vectors, shape (N, L, num_heads, num_q_points*3).
        """
        N, L = residue_feat.size()[:2]
        
        h = fn(residue_feat).view(N, L, self.num_heads * self.num_q_points, 3)
        h = local2global(R, coord, h)
        h = h.reshape(N, L, self.num_heads, -1)
        return h

    def _spatial_logits(self, R, coord, residue_feat):
        """
        Compute spatial attention logits based on frame transformations.

        Parameters
        ----------
        R : torch.Tensor
            Frame basis matrices, shape (N, L, 3, 3).
        coord : torch.Tensor
            Frame external coordinates, shape (N, L, 3).
        residue_feat : torch.Tensor
            Residue features, shape (N, L, residue_dim).

        Returns
        -------
        torch.Tensor
            Spatial attention logits, shape (N, L, L, num_heads).
        """
        q = self._compute_q_or_k(residue_feat, R, coord, fn=self.q_point_proj)
        k = self._compute_q_or_k(residue_feat, R, coord, fn=self.k_point_proj)
        
        sum_sq_dist = ((q.unsqueeze(2) - k.unsqueeze(1)) ** 2).sum(-1)
        gamma = F.softplus(self.spatial_coef)
        logits_spatial = sum_sq_dist * ((-1 * gamma * np.sqrt(2 / (9 * self.num_q_points))) / 2)
        
        return logits_spatial

    def _pair_aggregation(self, alpha, pair_feat):
        """
        Aggregate pair-wise features using attention weights.

        Parameters
        ----------
        alpha : torch.Tensor
            Attention weights, shape (N, L, L, num_heads).
        pair_feat : torch.Tensor
            Pair-wise features, shape (N, L, L, pair_dim).

        Returns
        -------
        torch.Tensor
            Aggregated pair-wise features, shape (N, L, num_heads * pair_dim).
        """
        feat_p2n = alpha.unsqueeze(-1) * pair_feat.unsqueeze(-2)
        feat_p2n = feat_p2n.sum(dim=2)
        
        return feat_p2n.reshape(pair_feat.size(0), pair_feat.size(1), -1)

    def _node_aggregation(self, alpha, residue_feat):
        """
        Aggregate node-wise features using attention weights.

        Parameters
        ----------
        alpha : torch.Tensor
            Attention weights, shape (N, L, L, num_heads).
        residue_feat : torch.Tensor
            Residue features, shape (N, L, residue_dim).

        Returns
        -------
        torch.Tensor
            Aggregated node-wise features, shape (N, L, num_heads * v_dim).
        """
        v = self.v_proj(residue_feat).view(residue_feat.size(0), residue_feat.size(1), self.num_heads, self.v_dim)
        feat_node = alpha.unsqueeze(-1) * v.unsqueeze(1)
        feat_node = feat_node.sum(dim=2)
        
        return feat_node.reshape(residue_feat.size(0), residue_feat.size(1), -1)


    def _spatial_aggregation(self, alpha, R, coord, residue_feat):
        """
        Aggregate spatial features using attention weights.

        Parameters
        ----------
        alpha : torch.Tensor
            Attention weights, shape (N, L, L, num_heads).
        R : torch.Tensor
            Frame basis matrices, shape (N, L, 3, 3).
        coord : torch.Tensor
            Frame external coordinates, shape (N, L, 3).
        residue_feat : torch.Tensor
            Residue features, shape (N, L, residue_dim).

        Returns
        -------
        torch.Tensor
            Aggregated spatial features, shape (N, L, num_heads * (points + direction + distance)).
        """
        # Project residue features to value points in global frame
        v_points = self.v_point_proj(residue_feat).view(residue_feat.size(0), residue_feat.size(1), self.num_heads * self.num_v_points, 3)
        
        # Convert local to global coordinates
        v_points = local2global(R, coord, v_points.reshape(residue_feat.size(0), residue_feat.size(1), self.num_heads, self.num_v_points, 3))
        
        # Attention-weighted aggregation of spatial features
        agg_points = alpha.reshape(residue_feat.size(0), residue_feat.size(1), residue_feat.size(1), self.num_heads, 1, 1) * v_points.unsqueeze(1)
        agg_points = agg_points.sum(dim=2)
        
        # Convert global back to local coordinates
        feat_points = global2local(R, coord, agg_points)
        
        # Calculate distance and direction features
        feat_distance = feat_points.norm(dim=-1)
        feat_direction = normalize_vector(feat_points, dim=-1, eps=1e-4)
        
        # Concatenate points, distance, and direction features
        feat_spatial = torch.cat([
            feat_points.reshape(residue_feat.size(0), residue_feat.size(1), -1),
            feat_distance.reshape(residue_feat.size(0), residue_feat.size(1), -1),
            feat_direction.reshape(residue_feat.size(0), residue_feat.size(1), -1)
        ], dim=-1)
        
        return feat_spatial

        
class VarianceSchedule(nn.Module):
    """
    Variance schedule module to compute alphas, betas, and sigmas used in diffusion processes.

    Parameters
    ----------
    num_steps : int, optional
        Number of time steps for the variance schedule. Default is 100.
    s : float, optional
        Smoothing factor for variance schedule calculation. Default is 0.01.
    """
    def __init__(self, num_steps=100, s=0.01):
        super().__init__()

        T = num_steps
        t = torch.arange(0, num_steps+1, dtype=torch.float)

        # Compute alphas and alpha_bars based on cosine schedule
        ft = (torch.cos((np.pi / 2) * (t / T + s) / (1 + s)) ** 2)
        alpha_bars = ft / ft[0]

        # Compute betas from alpha_bars
        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = torch.cat([torch.zeros([1]), betas], dim=0).clamp_max(0.999)

        # Compute sigmas using alpha_bars and betas
        sigmas = torch.zeros_like(betas)
        for i in range(1, betas.size(0)):
            sigmas[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas = torch.sqrt(sigmas)

        # Store parameters as buffers
        self.betas = torch.nn.Parameter(data=betas, requires_grad=False)
        self.alphas = torch.nn.Parameter(data=1 - betas, requires_grad=False)
        self.alpha_bars = torch.nn.Parameter(data=alpha_bars, requires_grad=False)
        self.sigmas = torch.nn.Parameter(data=sigmas, requires_grad=False)


class ApproxAngularDistribution(nn.Module):
    """
    Approximate angular distribution using either histograms or Gaussian approximation
    depending on the standard deviation.

    Parameters
    ----------
    stddevs : list of float
        List of standard deviations for different distributions.
    std_threshold : float, optional
        Threshold to decide whether to use histogram approximation or Gaussian sampling.
        Default is 0.1.
    num_bins : int, optional
        Number of bins for the histogram. Default is 8192.
    num_iters : int, optional
        Number of iterations for the PDF approximation. Default is 1024.
    """
    def __init__(self, stddevs, std_threshold=0.1, num_bins=8192, num_iters=1024):
        super().__init__()
        self.std_threshold = std_threshold
        self.num_bins = num_bins
        self.num_iters = num_iters

        # Register standard deviations and compute mask for Gaussian approximation
        self.register_buffer('stddevs', torch.FloatTensor(stddevs))
        self.register_buffer('approx_mask', self.stddevs <= std_threshold)
        
        # Precompute histograms for angular distributions
        self._precompute_histograms()

    def sample(self, std_idx):
        """
        Sample angles from the approximate distribution given a standard deviation index.

        Parameters
        ----------
        std_idx : torch.Tensor
            Tensor of indices representing different standard deviations.

        Returns
        -------
        torch.Tensor
            Sampled angles based on the distribution, shaped according to input.
        """
        size = std_idx.size()
        std_idx = std_idx.flatten()

        # Sample from histogram
        prob = self.hist[std_idx]
        bin_idx = torch.multinomial(prob[:, :-1], num_samples=1).squeeze(-1)
        bin_start = self.X[std_idx, bin_idx]
        bin_width = self.X[std_idx, bin_idx + 1] - self.X[std_idx, bin_idx]
        samples_from_hist = bin_start + torch.rand_like(bin_start) * bin_width

        # Sample from Gaussian approximation
        std = self.stddevs[std_idx]
        mu = 2 * std
        samples_from_gauss = mu + torch.randn_like(mu) * std
        samples_from_gauss = samples_from_gauss.abs() % math.pi

        # Choose between histogram or Gaussian samples based on mask
        gauss_mask = self.approx_mask[std_idx]
        samples = torch.where(gauss_mask, samples_from_gauss, samples_from_hist)

        return samples.reshape(size)

    def _precompute_histograms(self):
        """
        Precompute histograms for each standard deviation in `stddevs`.
        """
        X, Y = [], []

        # Compute histogram for each standard deviation
        for std in self.stddevs:
            std = std.item()
            x = torch.linspace(0, math.pi, self.num_bins)
            y = self._pdf(x, std, self.num_iters)
            y = torch.nan_to_num(y).clamp_min(0)
            X.append(x)
            Y.append(y)

        # Register buffers for histogram bins and probabilities
        self.register_buffer('X', torch.stack(X, dim=0))
        self.register_buffer('hist', torch.stack(Y, dim=0))

    @staticmethod
    def _pdf(x, std, num_iters):
        """
        Compute the probability density function for angular distribution.

        Parameters
        ----------
        x : torch.Tensor
            Input angles.
        std : float
            Standard deviation of the distribution.
        num_iters : int
            Number of iterations for the approximation.

        Returns
        -------
        torch.Tensor
            Probability densities for input angles.
        """
        x = x[:, None]
        c = ((1 - torch.cos(x)) / math.pi)
        l = torch.arange(0, num_iters)[None, :]
        a = (2 * l + 1) * torch.exp(-l * (l + 1) * (std ** 2))
        b = (torch.sin((l + 0.5) * x) + 1e-6) / (torch.sin(x / 2) + 1e-6)
        f = (c * a * b).sum(dim=-1)
        return f


class RotationTransition(nn.Module):
    """
    Handles rotation-based transitions for forward and reverse diffusion processes.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model parameters like number of steps and device.
    """
    def __init__(self, config):
        super(RotationTransition, self).__init__()

        self.config = config
        self.num_steps = config['num_steps']
        self.var_schedule = VarianceSchedule(num_steps=self.num_steps, s=config['ns_bias']).to(config["device"])

        # Forward diffusion angular distribution
        c1 = torch.sqrt(1 - self.var_schedule.alpha_bars)
        self.angular_distribution_forward = ApproxAngularDistribution(c1.tolist())

        # Backward diffusion angular distribution
        sigmas = self.var_schedule.sigmas
        self.angular_distribution_backward = ApproxAngularDistribution(sigmas.tolist())

    def add_noise(self, v0, generation_mask, t):
        """
        Adds noise to rotations during the forward diffusion process.

        Parameters
        ----------
        v0 : torch.Tensor
            Initial SO(3) vectors, shape (N, L, 3).
        generation_mask : torch.Tensor
            Mask for generation, shape (N, L).
        t : torch.Tensor
            Time step, shape (N,).

        Returns
        -------
        torch.Tensor
            Noisy SO(3) vectors, shape (N, L, 3).
        torch.Tensor
            Noise applied in SO(3), shape (N, L, 3).
        """
        N, L = generation_mask.size()
        alpha_bar = self.var_schedule.alpha_bars[t]

        # Compute scaling factors
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)

        # Sample rotation noise from IGso3
        e_scaled = randn_so3(t[:, None].expand(N, L), self.angular_distribution_forward, device=t.device)
        E_scaled = so3_vec2rotation(e_scaled)

        # Apply scaling to the true rotation and add noise
        R0_scaled = so3_vec2rotation(c0 * v0)
        R_noisy = R0_scaled @ E_scaled
        v_noisy = rotation_to_so3vec(R_noisy)

        # Apply the generation mask
        v_noisy = torch.where(generation_mask[..., None].expand_as(v0), v_noisy, v0)

        return v_noisy, e_scaled

    def denoise(self, v_t, v_tm1, generation_mask, t):
        """
        Denoises rotations during the reverse diffusion process.

        Parameters
        ----------
        v_t : torch.Tensor
            Noisy SO(3) vectors at time t, shape (N, L, 3).
        v_tm1 : torch.Tensor
            Predicted SO(3) vectors at time t-1, shape (N, L, 3).
        generation_mask : torch.Tensor
            Mask for generation, shape (N, L).
        t : torch.Tensor
            Time step, shape (N,).

        Returns
        -------
        torch.Tensor
            Denoised SO(3) vectors, shape (N, L, 3).
        """
        N, L = generation_mask.size()

        # Sample rotation noise for the backward diffusion
        e = randn_so3(t[:, None].expand(N, L), self.angular_distribution_backward, device=t.device)
        e = torch.where((t > 1)[:, None, None].expand(N, L, 3), e, torch.zeros_like(e))

        # Apply rotation and denoise
        E = so3_vec2rotation(e)
        R_tm1 = so3_vec2rotation(v_tm1)
        R_tm1 = R_tm1 @ E
        v_tm1 = rotation_to_so3vec(R_tm1)

        # Apply the generation mask
        v_tm1 = torch.where(generation_mask[..., None].expand_as(v_tm1), v_tm1, v_t)

        return v_tm1


class PositionTransition(nn.Module):
    """
    Handles position-based transitions for forward and reverse diffusion processes.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model parameters like number of steps and device.
    """
    def __init__(self, config):
        super(PositionTransition, self).__init__()

        self.config = config
        self.num_steps = config['num_steps']
        self.var_schedule = VarianceSchedule(num_steps=self.num_steps).to(config["device"])
        self.alphas = self.var_schedule.alphas
        self.alpha_bars = self.var_schedule.alpha_bars
        self.sigmas = self.var_schedule.sigmas

    def add_noise(self, p0, generation_mask, t):
        """
        Adds noise to positions during the forward diffusion process.

        Parameters
        ----------
        p0 : torch.Tensor
            Initial positions, shape (N, L, 3).
        generation_mask : torch.Tensor
            Mask for generation, shape (N, L).
        t : torch.Tensor
            Time step, shape (N,).

        Returns
        -------
        torch.Tensor
            Noisy positions, shape (N, L, 3).
        torch.Tensor
            Noise applied to positions, shape (N, L, 3).
        """
        alpha_bar = self.var_schedule.alpha_bars[t]

        # Compute scaling factors
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)

        eps = torch.randn_like(p0)
        eps_input = torch.randn_like(p0) if self.config["input_error"] else torch.zeros_like(p0)
        gamma = 0.1 * torch.ones_like(t).view(-1, 1, 1)

        # Add noise to positions
        p_noisy = c0 * p0 + c1 * (eps + gamma * eps_input)
        p_noisy = torch.where(generation_mask[..., None].expand_as(p0), p_noisy, p0)

        return p_noisy, eps

    def denoise(self, p_t, eps_hat, generation_mask, t):
        """
        Denoises positions during the reverse diffusion process.

        Parameters
        ----------
        p_t : torch.Tensor
            Noisy positions at time t, shape (N, L, 3).
        eps_hat : torch.Tensor
            Predicted noise to remove, shape (N, L, 3).
        generation_mask : torch.Tensor
            Mask for generation, shape (N, L).
        t : torch.Tensor
            Time step, shape (N,).

        Returns
        -------
        torch.Tensor
            Denoised positions, shape (N, L, 3).
        """
        alpha = self.alphas[t].clamp_min(self.alphas[-2])
        alpha_bar = self.alpha_bars[t]
        sigma = self.sigmas[t].view(-1, 1, 1)

        # Compute scaling factors for denoising
        c0 = (1.0 / torch.sqrt(alpha + 1e-8)).view(-1, 1, 1)
        c1 = ((1.0 - alpha) / torch.sqrt(1 - alpha_bar + 1e-8)).view(-1, 1, 1)

        # Sample random noise
        z = torch.where((t > 1)[:, None, None].expand_as(p_t), torch.randn_like(p_t), torch.zeros_like(p_t))

        # Compute denoised positions
        p_tm1 = c0 * (p_t - c1 * eps_hat) + sigma * z
        p_tm1 = torch.where(generation_mask[..., None].expand_as(p_t), p_tm1, p_t)

        return p_tm1


class SequenceTransition(nn.Module):
    """
    Handles sequence-based transitions for forward and reverse diffusion processes.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model parameters like number of steps and device.
    """
    def __init__(self, config):
        super(SequenceTransition, self).__init__()

        self.config = config
        self.num_steps = config['num_steps']
        self.K = config['num_aa_types']
        self.var_schedule = VarianceSchedule(num_steps=self.num_steps).to(config["device"])
        self.alphas = self.var_schedule.alphas
        self.alpha_bars = self.var_schedule.alpha_bars

    @staticmethod
    def _sample(c):
        """
        Samples sequences based on categorical probabilities.

        Parameters
        ----------
        c : torch.Tensor
            Input probabilities for each category, shape (N, L, K).

        Returns
        -------
        torch.Tensor
            Sampled sequences, shape (N, L).
        """
        N, L, K = c.size()
        c = c.view(N * L, K) + 1e-8
        seqs = torch.multinomial(c, 1).view(N, L)
        return seqs

    def posterior(self, x_t, x_0, t):
        """
        Computes the posterior probability at time step t-1.

        Parameters
        ----------
        x_t : torch.Tensor
            Sequence at time t, either in categorical form (N, L) or probability form (N, L, K).
        x_0 : torch.Tensor
            Original sequence at time t=0, either in categorical form (N, L) or probability form (N, L, K).
        t : torch.Tensor
            Time step, shape (N,).

        Returns
        -------
        torch.Tensor
            Posterior probability, shape (N, L, K).
        """
        K = self.K

        c_0 = x_0 if x_0.dim() == 3 else self.clampped_one_hot(x_0, num_classes=K).float()
        c_t = x_t if x_t.dim() == 3 else self.clampped_one_hot(x_t, num_classes=K).float()

        alpha = self.alphas[t][:, None, None]
        alpha_bar = self.alpha_bars[torch.clamp(t - 1, min=0)][:, None, None]

        theta = (alpha * c_t + (1 - alpha) / K) * (alpha_bar * c_0 + (1 - alpha_bar) / K)
        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-8)

        return theta

    def clampped_one_hot(self, x, num_classes):
        """
        Converts a tensor to a clamped one-hot encoding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with categories, shape (N, L).
        num_classes : int
            Number of classes for one-hot encoding.

        Returns
        -------
        torch.Tensor
            One-hot encoded tensor, shape (N, L, num_classes).
        """
        mask = (x >= 0) & (x < num_classes)
        x = x.clamp(min=0, max=num_classes - 1)
        y = F.one_hot(x, num_classes) * mask[..., None]
        return y

    def add_noise(self, x0, generation_mask, t):
        """
        Adds noise to sequences during the forward diffusion process.

        Parameters
        ----------
        x0 : torch.Tensor
            Original sequence at time t=0, shape (N, L).
        generation_mask : torch.Tensor
            Mask for generation, shape (N, L).
        t : torch.Tensor
            Time step, shape (N,).

        Returns
        -------
        torch.Tensor
            Noisy sequence probabilities, shape (N, L, K).
        torch.Tensor
            Sampled noisy sequence, shape (N, L).
        """
        N, L = x0.size()
        K = self.K

        c0 = self.clampped_one_hot(x0, num_classes=K).float()
        alpha_bar = self.alpha_bars[t][:, None, None]

        uniform_noise = 1 / K
        c_noisy = (alpha_bar * c0) + ((1 - alpha_bar) * uniform_noise)
        c_t = torch.where(generation_mask[..., None].expand(N, L, K), c_noisy, c0)
        x_t = self._sample(c_t)

        return c_t, x_t

    def denoise(self, x_t, c0_pred, generation_mask, t):
        """
        Denoises sequences during the reverse diffusion process.

        Parameters
        ----------
        x_t : torch.Tensor
            Noisy sequence at time t, shape (N, L).
        c0_pred : torch.Tensor
            Predicted sequence probabilities at time t=0, shape (N, L, K).
        generation_mask : torch.Tensor
            Mask for generation, shape (N, L).
        t : torch.Tensor
            Time step, shape (N,).

        Returns
        -------
        torch.Tensor
            Posterior probability at time t-1, shape (N, L, K).
        torch.Tensor
            Sampled sequence at time t-1, shape (N, L).
        """
        N, L = x_t.size()
        c_t = self.clampped_one_hot(x_t, num_classes=self.K).float()
        post = self.posterior(c_t, c0_pred, t=t)
        post = torch.where(generation_mask[..., None].expand(N, L, self.K), post, c_t)
        x_tm1 = self._sample(post)

        return post, x_tm1


class ResidueEmbedding(nn.Module):
    """
    Embeds residue-level features, including amino acid types, chain types, dihedral angles,
    and atom coordinates, into a fixed-dimensional embedding space.

    Parameters
    ----------
    residue_dim : int
        Dimension of the residue embedding.
    num_atoms : int
        Number of atoms to consider in each residue.
    max_aa_types : int, optional
        Maximum number of amino acid types (default is 22).
    max_chain_types : int, optional
        Maximum number of chain types (default is 10).
    """
    def __init__(self, residue_dim, num_atoms, max_aa_types=22, max_chain_types=10):
        super(ResidueEmbedding, self).__init__()
        
        self.residue_dim = residue_dim
        self.num_atoms = num_atoms
        self.max_aa_types = max_aa_types
        
        # Embeddings for amino acids, chain types, and dihedral angles
        self.aa_emb = nn.Embedding(max_aa_types, residue_dim)
        self.chain_emb = nn.Embedding(max_chain_types, residue_dim, padding_idx=0)
        self.dihedral_emb = DihedralEncoding()

        # Input dimension for the MLP layer
        input_dim = residue_dim + max_aa_types * num_atoms * 3 + self.dihedral_emb.get_dim() + residue_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2 * residue_dim), nn.ReLU(), 
            nn.Linear(2 * residue_dim, residue_dim), nn.ReLU(), 
            nn.Linear(residue_dim, residue_dim), nn.ReLU(), 
            nn.Linear(residue_dim, residue_dim)
        )

    def forward(self, aa, res_nb, fragment_type, pos_atoms, mask_atoms, structure_mask=None, sequence_mask=None, generation_mask_bar=None):
        """
        Forward pass for residue embedding.

        Parameters
        ----------
        aa : torch.Tensor
            Amino acid types, shape (N, L).
        res_nb : torch.Tensor
            Residue numbers, shape (N, L).
        fragment_type : torch.Tensor
            Chain fragment types, shape (N, L).
        pos_atoms : torch.Tensor
            Atom coordinates, shape (N, L, A, 3).
        mask_atoms : torch.Tensor
            Atom masks, shape (N, L, A).
        structure_mask : torch.Tensor, optional
            Mask for known structures, shape (N, L).
        sequence_mask : torch.Tensor, optional
            Mask for known amino acids, shape (N, L).
        generation_mask_bar : torch.Tensor, optional
            Mask for generated amino acids, shape (N, L).

        Returns
        -------
        torch.Tensor
            Residue embeddings, shape (N, L, residue_dim).
        """
        N, L = aa.size()

        # Mask for valid residues based on atoms
        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA]
        pos_atoms = pos_atoms[:, :, :self.num_atoms]
        mask_atoms = mask_atoms[:, :, :self.num_atoms]

        # 1. Chain embedding (N, L, residue_dim)
        chain_emb = self.chain_emb(fragment_type)

        # 2. Amino acid embedding (N, L, residue_dim)
        if sequence_mask is not None:
            aa = torch.where(sequence_mask, aa, torch.full_like(aa, fill_value=AALib.UNK))
        aa_emb = self.aa_emb(aa)

        # 3. Coordinate embedding (N, L, max_aa_types * num_atoms * 3)
        bb_center = pos_atoms[:, :, BBHeavyAtom.CA]
        R = get_3d_basis(center=bb_center, p1=pos_atoms[:, :, BBHeavyAtom.C], p2=pos_atoms[:, :, BBHeavyAtom.N])
        local_coords = global2local(R, bb_center, pos_atoms)
        local_coords = torch.where(mask_atoms[:, :, :, None].expand_as(local_coords), local_coords, torch.zeros_like(local_coords))

        # Expand amino acid embedding and apply mask
        aa_expand = aa[:, :, None, None, None].expand(N, L, self.max_aa_types, self.num_atoms, 3)
        aa_range = torch.arange(0, self.max_aa_types)[None, None, :, None, None].expand(N, L, self.max_aa_types, self.num_atoms, 3).to(aa_expand)
        aa_expand_mask = (aa_expand == aa_range)
        local_coords_expand = local_coords[:, :, None, :, :].expand(N, L, self.max_aa_types, self.num_atoms, 3)
        local_coords = torch.where(aa_expand_mask, local_coords_expand, torch.zeros_like(local_coords_expand))
        local_coords = local_coords.reshape(N, L, self.max_aa_types * self.num_atoms * 3)

        if structure_mask is not None and structure_mask.dim() == 2:
            structure_mask = structure_mask.unsqueeze(-1)  # Expand to (N, L, 1)
            local_coords = local_coords * structure_mask

        # 4. Dihedral angle embedding (N, L, 39)
        bb_dihedral, mask_bb_dihedral = get_bb_dihedral_angles(pos_atoms, fragment_type, res_nb=res_nb, mask_residue=mask_residue)
        dihedral_emb = self.dihedral_emb(bb_dihedral[:, :, :, None])
        dihedral_emb = dihedral_emb * mask_bb_dihedral[:, :, :, None]
        dihedral_emb = dihedral_emb.reshape(N, L, -1)

        if structure_mask is not None:
            dihedral_mask = torch.logical_and(
                structure_mask.squeeze(-1),
                torch.logical_and(
                    torch.roll(structure_mask.squeeze(-1), shifts=+1, dims=1), 
                    torch.roll(structure_mask.squeeze(-1), shifts=-1, dims=1)
                )
            )
            dihedral_emb = dihedral_emb * dihedral_mask[:, :, None]

        # 5. Concatenate all features and apply mask
        all_features = torch.cat([aa_emb, local_coords, dihedral_emb, chain_emb], dim=-1)
        all_features = all_features * mask_residue[:, :, None].expand_as(all_features)

        # 6. Apply MLP to generate final embeddings
        out_features = self.mlp(all_features)
        out_features = out_features * mask_residue[:, :, None]

        return out_features


class PairEmbedding(nn.Module):
    """
    Embeds pairwise residue features including amino acid pairs, relative positions, atom-atom distances, and dihedral angles.

    Parameters
    ----------
    pair_dim : int
        Dimension of the pair embedding.
    num_atoms : int
        Number of atoms to consider for pairwise distance calculations.
    max_aa_types : int, optional
        Maximum number of amino acid types (default is 22).
    max_relpos : int, optional
        Maximum relative position (default is 32).
    """
    def __init__(self, pair_dim, num_atoms, max_aa_types=22, max_relpos=32):
        super(PairEmbedding, self).__init__()

        self.pair_dim = pair_dim
        self.num_atoms = num_atoms
        self.max_aa_types = max_aa_types
        self.max_relpos = max_relpos

        # Pair embedding, relative position embedding, and distance embedding
        self.aa_pair_emb = nn.Embedding(max_aa_types**2, pair_dim)
        self.relpos_emb = nn.Embedding(2 * max_relpos + 1, pair_dim)
        self.aapair_to_dist_coeff = nn.Embedding(max_aa_types**2, num_atoms**2)
        nn.init.zeros_(self.aapair_to_dist_coeff.weight)

        # Distance embedding and dihedral embedding
        self.dist_emb = nn.Sequential(nn.Linear(num_atoms**2, pair_dim), nn.ReLU(), nn.Linear(pair_dim, pair_dim), nn.ReLU())
        self.dihedral_emb = DihedralEncoding()
        dihedral_feature_dim = self.dihedral_emb.get_dim(num_dim=2)

        # MLP for final pair embedding
        all_features_dim = 3 * pair_dim + dihedral_feature_dim
        self.mlp = nn.Sequential(nn.Linear(all_features_dim, pair_dim), nn.ReLU(), nn.Linear(pair_dim, pair_dim), nn.ReLU(), nn.Linear(pair_dim, pair_dim))

    def forward(self, aa, res_nb, fragment_type, pos_atoms, mask_atoms, structure_mask=None, sequence_mask=None):
        """
        Forward pass for pairwise residue embedding.

        Parameters
        ----------
        aa : torch.Tensor
            Amino acid types, shape (N, L).
        res_nb : torch.Tensor
            Residue numbers, shape (N, L).
        fragment_type : torch.Tensor
            Chain fragment types, shape (N, L).
        pos_atoms : torch.Tensor
            Atom coordinates, shape (N, L, A, 3).
        mask_atoms : torch.Tensor
            Atom masks, shape (N, L, A).
        structure_mask : torch.Tensor, optional
            Mask for known structures, shape (N, L).
        sequence_mask : torch.Tensor, optional
            Mask for known amino acids, shape (N, L).

        Returns
        -------
        torch.Tensor
            Pairwise residue embeddings, shape (N, L, L, pair_dim).
        """
        N, L = aa.size()

        # Mask for valid residues
        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA]
        pos_atoms = pos_atoms[:, :, :self.num_atoms]
        mask_atoms = mask_atoms[:, :, :self.num_atoms]
        mask2d_pair = mask_residue[:, :, None] * mask_residue[:, None, :]

        # 1. Pairwise amino acid embedding
        if sequence_mask is not None:
            aa = torch.where(sequence_mask, aa, torch.full_like(aa, fill_value=AALib.UNK))
        aa_pair = self.max_aa_types * aa[:, :, None] + aa[:, None, :]
        aa_pair_emb = self.aa_pair_emb(aa_pair)

        # 2. Relative position embedding
        relative_pos = res_nb[:, :, None] - res_nb[:, None, :]
        relative_pos = torch.clamp(relative_pos, min=-self.max_relpos, max=self.max_relpos) + self.max_relpos
        relative_pos_emb = self.relpos_emb(relative_pos)
        mask2d_chain = (fragment_type[:, :, None] == fragment_type[:, None, :])
        relative_pos_emb = relative_pos_emb * mask2d_chain[:, :, :, None]

        # 3. Atom-atom distance embedding
        a2a_coords = pos_atoms[:, :, None, :, None] - pos_atoms[:, None, :, None, :]
        a2a_dist = torch.linalg.norm(a2a_coords, dim=-1)
        a2a_dist_nm = a2a_dist / 10.
        a2a_dist_nm = a2a_dist_nm.reshape(N, L, L, -1)
        coeff = F.softplus(self.aapair_to_dist_coeff(aa_pair))
        dist_rbf = torch.exp(-1.0 * coeff * a2a_dist_nm**2)
        mask2d_aa_pair = mask_atoms[:, :, None, :, None] * mask_atoms[:, None, :, None, :]
        mask2d_aa_pair = mask2d_aa_pair.reshape(N, L, L, -1)
        dist_emb = self.dist_emb(dist_rbf * mask2d_aa_pair)

        # 4. Dihedral angle embedding
        dihedral_angles = pairwise_dihedrals(pos_atoms)
        dihedral_emb = self.dihedral_emb(dihedral_angles)

        # Apply structure mask to avoid data leakage
        if structure_mask is not None and structure_mask.dim() == 2:
            structure_mask = structure_mask.unsqueeze(-1)
            dist_emb = dist_emb * structure_mask[:, :, :, None]
            dihedral_emb = dihedral_emb * structure_mask[:, :, :, None]

        # 5. Combine all features
        all_features = torch.cat([aa_pair_emb, relative_pos_emb, dist_emb, dihedral_emb], dim=-1)
        all_features = all_features * mask2d_pair[:, :, :, None].expand_as(all_features)

        # 6. Apply MLP for final pairwise embedding
        out = self.mlp(all_features)
        out = out * mask2d_pair[:, :, :, None]

        return out


class DihedralEncoding(nn.Module):
    """
    Dihedral angle encoding using sinusoidal and cosinusoidal transformations.

    Parameters
    ----------
    num_freq_bands : int, optional
        Number of frequency bands for encoding (default is 3).
    """
    def __init__(self, num_freq_bands=3):
        super().__init__()

        self.num_freq_bands = num_freq_bands
        self.register_buffer('freq_bands', torch.FloatTensor([i + 1 for i in range(num_freq_bands)] + [1. / (i + 1) for i in range(num_freq_bands)]))

    def forward(self, x):
        """
        Forward pass for dihedral encoding.

        Parameters
        ----------
        x : torch.Tensor
            Backbone dihedral angles, shape (B, L, 3, 1).

        Returns
        -------
        torch.Tensor
            Encoded dihedral angles, shape (B, L, 3, -1).
        """
        shape = list(x.shape[:-1]) + [-1]
        x = x.unsqueeze(-1)
        angle_emb = torch.cat([x, torch.sin(x * self.freq_bands), torch.cos(x * self.freq_bands)], dim=-1)
        return angle_emb.reshape(shape)

    def get_dim(self, num_dim=3):
        """
        Returns the dimension of the dihedral encoding.

        Parameters
        ----------
        num_dim : int, optional
            Number of dihedral angles (default is 3).

        Returns
        -------
        int
            Dimension of the dihedral encoding.
        """
        return num_dim * (1 + 2 * 2 * self.num_freq_bands)