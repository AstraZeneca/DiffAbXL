"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: Library of utils needed for training diffusion model.
"""

import logging
import torch
import numpy as np
import torch.nn.functional as F
from torch.profiler import record_function
from inspect import isfunction
from easydict import EasyDict

import torch
import torch.nn.functional as F
import numpy as np

def clampped_one_hot(x, num_classes):
    """
    Computes a clamped one-hot encoding of the input tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor containing class indices, shape (N, L).
    num_classes : int
        The number of classes for one-hot encoding.

    Returns
    -------
    torch.Tensor
        One-hot encoded tensor with the same batch size, shape (N, L, C).
    """
    mask = (x >= 0) & (x < num_classes)
    x = x.clamp(min=0, max=num_classes - 1)
    y = F.one_hot(x, num_classes) * mask[..., None]
    return y


def move2device(obj, device='cuda'):
    """
    Moves a given object (Tensor, list, tuple, dict) to a specific device (CPU or GPU).

    Parameters
    ----------
    obj : any
        Object to move to the specified device. Can be a torch.Tensor, list, tuple, or dict.
    device : str, optional
        The target device (default is 'cuda').

    Returns
    -------
    any
        Object moved to the specified device.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move2device(o, device=device) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(move2device(o, device=device) for o in obj)
    elif isinstance(obj, dict):
        return {k: move2device(v, device=device) for k, v in obj.items()}
    return obj


def inf_iterator(iterable):
    """
    Creates an infinite iterator over the provided iterable.

    Parameters
    ----------
    iterable : iterable
        The iterable to create an infinite loop over.

    Yields
    ------
    any
        Next item from the iterable.
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Computes the KL divergence between two Gaussian distributions.

    Parameters
    ----------
    mean1, logvar1, mean2, logvar2 : torch.Tensor
        Mean and log-variance tensors for both distributions.

    Returns
    -------
    torch.Tensor
        KL divergence between the two distributions.
    """
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(mean1.device)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function (CDF) of the standard normal distribution.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Approximated CDF values.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Computes the log-likelihood of a Gaussian distribution discretizing to a given image.

    Parameters
    ----------
    x : torch.Tensor
        The target images, assumed to be uint8 values rescaled to the range [-1, 1].
    means : torch.Tensor
        The Gaussian mean tensor.
    log_scales : torch.Tensor
        The Gaussian log stddev tensor.

    Returns
    -------
    torch.Tensor
        Log probabilities (in nats).
    """
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)

    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
        x < -0.999, log_cdf_plus, torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12)))
    )
    return log_probs


def sum_except_batch(x, num_dims=1):
    """
    Sums all dimensions except the first `num_dims` batch dimensions.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch_size, ...).
    num_dims : int, optional
        Number of batch dimensions, by default 1.

    Returns
    -------
    torch.Tensor
        Tensor with batch dimensions retained, shape (batch_size,).
    """
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def mean_flat(tensor):
    """
    Computes the mean across all non-batch dimensions.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Mean across non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, tensor.ndim)))


def ohe_to_categories(ohe, K):
    """
    Converts one-hot encoded tensor to categorical indices.

    Parameters
    ----------
    ohe : torch.Tensor
        One-hot encoded tensor, shape (N, sum(K)).
    K : list
        List of class sizes for each categorical variable.

    Returns
    -------
    torch.Tensor
        Categorical indices, shape (N, len(K)).
    """
    K = torch.from_numpy(K)
    indices = torch.cat([torch.zeros((1,)), K.cumsum(dim=0)], dim=0).int().tolist()
    res = [ohe[:, indices[i]:indices[i+1]].argmax(dim=1) for i in range(len(indices) - 1)]
    return torch.stack(res, dim=1)


def log_1_min_a(a):
    """
    Computes the log of `1 - exp(a)` safely to avoid numerical instability.

    Parameters
    ----------
    a : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Log of `1 - exp(a)`.
    """
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    """
    Computes the log of the sum of two exponentials in a numerically stable way.

    Parameters
    ----------
    a, b : torch.Tensor
        Input tensors.

    Returns
    -------
    torch.Tensor
        Logarithm of the sum of exponentials of the inputs.
    """
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def exists(x):
    """
    Checks if a variable is not None.

    Parameters
    ----------
    x : any
        Input variable.

    Returns
    -------
    bool
        True if x is not None, False otherwise.
    """
    return x is not None


def extract(a, t, x_shape):
    """
    Extracts values from tensor `a` at index `t` and reshapes them to match `x_shape`.

    Parameters
    ----------
    a : torch.Tensor
        Input tensor from which values are extracted.
    t : torch.Tensor
        Indices at which to extract values.
    x_shape : tuple
        Shape of the output tensor.

    Returns
    -------
    torch.Tensor
        Extracted and reshaped tensor.
    """
    out = a.gather(-1, t.to(a.device))
    while len(out.shape) < len(x_shape):
        out = out[..., None]
    return out.expand(x_shape)


def default(val, d):
    """
    Returns `val` if it is not None, otherwise returns `d`.

    Parameters
    ----------
    val : any
        The value to check.
    d : any
        The default value to return if `val` is None.

    Returns
    -------
    any
        `val` if not None, otherwise `d`.
    """
    return val if exists(val) else d() if callable(d) else d


def log_categorical(log_x_start, log_prob):
    """
    Computes the log-likelihood for categorical data.

    Parameters
    ----------
    log_x_start : torch.Tensor
        Logarithm of the starting values.
    log_prob : torch.Tensor
        Logarithm of the probabilities.

    Returns
    -------
    torch.Tensor
        Log-likelihood for categorical data.
    """
    return (log_x_start.exp() * log_prob).sum(dim=1)


def index_to_log_onehot(x, num_classes):
    """
    Converts index tensor to a log one-hot encoded tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input index tensor.
    num_classes : int
        Number of classes for one-hot encoding.

    Returns
    -------
    torch.Tensor
        Log one-hot encoded tensor.
    """
    onehots = [F.one_hot(x[:, i], num_classes[i]) for i in range(len(num_classes))]
    x_onehot = torch.cat(onehots, dim=1)
    return torch.log(x_onehot.float().clamp(min=1e-30))


def log_sum_exp_by_classes(x, slices):
    """
    Computes the log-sum-exp over specific class slices in a tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    slices : list of slice
        List of slices to sum over.

    Returns
    -------
    torch.Tensor
        Tensor with log-sum-exp over the specified slices.
    """
    res = torch.zeros_like(x)
    for ixs in slices:
        res[:, ixs] = torch.logsumexp(x[:, ixs], dim=1, keepdim=True)
    return res


@torch.jit.script
def log_sub_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the log of the difference of two exponentials in a numerically stable way.

    Parameters
    ----------
    a, b : torch.Tensor
        Input tensors.

    Returns
    -------
    torch.Tensor
        Logarithm of the difference of exponentials of the inputs.
    """
    m = torch.maximum(a, b)
    return torch.log(torch.exp(a - m) - torch.exp(b - m)) + m


@torch.jit.script
def sliced_logsumexp(x, slices):
    """
    Computes the log-sum-exp over specific slices of the input tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    slices : list
        List of slices defining the sections to compute log-sum-exp.

    Returns
    -------
    torch.Tensor
        Tensor with log-sum-exp applied to the specified slices.
    """
    lse = torch.logcumsumexp(F.pad(x, [1, 0, 0, 0], value=-float('inf')), dim=-1)
    slice_starts, slice_ends = slices[:-1], slices[1:]
    slice_lse = log_sub_exp(lse[:, slice_ends], lse[:, slice_starts])
    return torch.repeat_interleave(slice_lse, slice_ends - slice_starts, dim=-1)


def log_onehot_to_index(log_x):
    """
    Converts a log one-hot tensor to categorical indices.

    Parameters
    ----------
    log_x : torch.Tensor
        Log one-hot encoded tensor.

    Returns
    -------
    torch.Tensor
        Categorical index tensor.
    """
    return log_x.argmax(dim=1)


class FoundNANsError(BaseException):
    """
    Exception raised when NaNs are found during sampling.

    Attributes
    ----------
    message : str
        Error message to display when exception is raised.
    """
    def __init__(self, message='Found NaNs during sampling.'):
        super().__init__(message)
