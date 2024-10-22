"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: Sample script to compute log-likehood.
"""

import torch


def compute_log_likelihood(sequence_tokens_list, posterior_list, parent_aa_list=None):
    """
    Compute the log-likelihood for each sequence in the batch.

    Args:
        sequence_tokens_list (list of Tensors): List of tensors of size (batch_size_i, sequence_length) containing sequence tokens.
        posterior_list (list of Tensors): List of tensors of size (batch_size_i, sequence_length, 20) containing posterior probabilities over amino acids.
        parent_aa_list (list of Tensors, optional): List of tensors containing parent amino acid tokens.

    Returns:
        log_likelihoods (Tensor): Tensor of log-likelihood values for the batch.
        log_likelihood_per_position (Tensor): Tensor of log-likelihood per position.
    """
    # Concatenate the list of tensors along the batch dimension
    sequence_tokens = torch.cat(sequence_tokens_list, dim=0)
    posterior = torch.cat(posterior_list, dim=0)

    # Compute log probabilities from posterior
    log_posterior = torch.log(posterior + 1e-9)  # Avoid log(0) by adding a small epsilon
    log_posterior = log_posterior.sum(0).unsqueeze(0).repeat(sequence_tokens.size(0), 1, 1)

    # Gather the log probabilities corresponding to the actual sequence tokens
    log_likelihood_per_position = torch.gather(
        log_posterior, dim=2, index=sequence_tokens.unsqueeze(-1)
    ).squeeze(-1)

    if parent_aa_list is not None and len(parent_aa_list) > 0:
        parent_aa_tokens = torch.cat(parent_aa_list, dim=0)
        parent_log_likelihood_per_position = torch.gather(
            log_posterior, dim=2, index=parent_aa_tokens.unsqueeze(-1)
        ).squeeze(-1)
        log_likelihood_per_position = log_likelihood_per_position - parent_log_likelihood_per_position

    # Sum the log-likelihood over the sequence length to get the total log-likelihood for each sequence
    log_likelihoods = log_likelihood_per_position.sum(dim=1)

    return log_likelihoods, log_likelihood_per_position
