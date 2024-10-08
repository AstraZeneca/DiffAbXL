"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: Library of loss functions.
"""

import numpy as np
import torch
import torch.nn.functional as F


def rotation_loss(R_pred, R_true):
    """
    Computes the loss between predicted and true rotation matrices using cosine embedding loss.

    Parameters
    ----------
    R_pred : torch.Tensor
        Predicted rotation matrices, shape (N, L, 3, 3).
    R_true : torch.Tensor
        True rotation matrices, shape (N, L, 3, 3).

    Returns
    -------
    torch.Tensor
        Per-matrix loss, shape (N, L), representing the loss for each predicted rotation.
    """
    size = list(R_pred.shape[:-2])  # Batch dimensions (N, L)
    ncol = R_pred.numel() // 3  # Number of columns after reshaping

    # Transpose and reshape the predicted and true rotation matrices
    RT_pred = R_pred.transpose(-2, -1).reshape(ncol, 3)
    RT_true = R_true.transpose(-2, -1).reshape(ncol, 3)

    # Compute the cosine embedding loss
    ones = torch.ones([ncol], dtype=torch.long, device=R_pred.device)
    loss = F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction='none')
    
    # Reshape loss and sum over the last dimension
    loss = loss.reshape(size + [3]).sum(dim=-1)  # Shape (N, L)
    
    return loss


def sum_weighted_losses(losses, weights):
    """
    Sums the weighted losses.

    Parameters
    ----------
    losses : dict
        Dictionary of scalar tensors representing different losses.
    weights : dict
        Dictionary of weights for each loss.

    Returns
    -------
    torch.Tensor
        Weighted sum of losses.
    """
    loss = 0
    for key in losses.keys():
        if weights is None:
            loss += losses[key]
        else:
            loss += weights[key] * losses[key]
    return loss
