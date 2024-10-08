"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: Library of utils for geometry
"""

import numpy as np
import torch
import torch.nn.functional as F
from utils.protein_constants import CDR, AALib, restype_to_heavyatom_names, BBHeavyAtom, backbone_atom_coordinates_tensor, bb_oxygen_coordinate_tensor


def compose_rotation_and_translation(R1, t1, R2, t2):
    """
    Compose two rotations and translations.

    Parameters
    ----------
    R1 : torch.Tensor
        Frame basis, shape (N, L, 3, 3).
    t1 : torch.Tensor
        Translation, shape (N, L, 3).
    R2 : torch.Tensor
        Rotation to be applied, shape (N, L, 3, 3).
    t2 : torch.Tensor
        Translation to be applied, shape (N, L, 3).

    Returns
    -------
    R_new : torch.Tensor
        Composed rotation, shape (N, L, 3, 3).
    t_new : torch.Tensor
        Composed translation, shape (N, L, 3).
    """
    R_new = torch.matmul(R1, R2)
    t_new = torch.matmul(R1, t2.unsqueeze(-1)).squeeze(-1) + t1
    return R_new, t_new


def compose_rotation_translation(Ts):
    """
    Iteratively compose a list of rotations and translations.

    Parameters
    ----------
    Ts : list of tuples
        List of (R, t) pairs where R is the rotation matrix and t is the translation vector.

    Returns
    -------
    tuple
        Composed (R, t) pair.
    """
    while len(Ts) >= 2:
        R1, t1 = Ts[-2]
        R2, t2 = Ts[-1]
        T_next = compose_rotation_and_translation(R1, t1, R2, t2)
        Ts = Ts[:-2] + [T_next]
    return Ts[0]


def normalize_vector(v, dim, eps=1e-6):
    """
    Normalize a vector along the specified dimension.

    Parameters
    ----------
    v : torch.Tensor
        Input vector to be normalized.
    dim : int
        Dimension along which to normalize.
    eps : float, optional
        Small epsilon value to avoid division by zero (default is 1e-6).

    Returns
    -------
    torch.Tensor
        Normalized vector.
    """
    return v / (torch.linalg.norm(v, ord=2, dim=dim, keepdim=True) + eps)


def project_v2v(v, e, dim):
    """
    Project vector `v` onto vector `e`.

    Parameters
    ----------
    v : torch.Tensor
        Input vector, shape (N, L, 3).
    e : torch.Tensor
        Vector onto which `v` will be projected, shape (N, L, 3).
    dim : int
        Dimension along which to compute the projection.

    Returns
    -------
    torch.Tensor
        Projected vector, shape (N, L, 3).
    """
    return (e * v).sum(dim=dim, keepdim=True) * e


def get_3d_basis(center, p1, p2):
    """
    Compute a 3D orthogonal basis given three points.

    Parameters
    ----------
    center : torch.Tensor
        Central point, usually the position of C_alpha, shape (N, L, 3).
    p1 : torch.Tensor
        First point, usually the position of C, shape (N, L, 3).
    p2 : torch.Tensor
        Second point, usually the position of N, shape (N, L, 3).

    Returns
    -------
    torch.Tensor
        Orthogonal basis matrix, shape (N, L, 3, 3).
    """
    v1 = p1 - center
    e1 = normalize_vector(v1, dim=-1)

    v2 = p2 - center
    u2 = v2 - project_v2v(v2, e1, dim=-1)
    e2 = normalize_vector(u2, dim=-1)

    e3 = torch.cross(e1, e2, dim=-1)

    return torch.cat([e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)], dim=-1)


def local2global(R, t, p):
    """
    Convert local coordinates to global coordinates.

    Parameters
    ----------
    R : torch.Tensor
        Rotation matrix, shape (N, L, 3, 3).
    t : torch.Tensor
        Translation vector, shape (N, L, 3).
    p : torch.Tensor
        Local coordinates, shape (N, L, ..., 3).

    Returns
    -------
    torch.Tensor
        Global coordinates, shape (N, L, ..., 3).
    """
    assert p.size(-1) == 3
    p_size = p.size()
    N, L = p_size[0], p_size[1]

    p = p.view(N, L, -1, 3).transpose(-1, -2)
    q = torch.matmul(R, p) + t.unsqueeze(-1)
    q = q.transpose(-1, -2).reshape(p_size)
    return q


def global2local(R, t, q):
    """
    Convert global coordinates to local coordinates.

    Parameters
    ----------
    R : torch.Tensor
        Rotation matrix, shape (N, L, 3, 3).
    t : torch.Tensor
        Translation vector, shape (N, L, 3).
    q : torch.Tensor
        Global coordinates, shape (N, L, ..., 3).

    Returns
    -------
    torch.Tensor
        Local coordinates, shape (N, L, ..., 3).
    """
    assert q.size(-1) == 3
    q_size = q.size()
    N, L = q_size[0], q_size[1]

    q = q.view(N, L, -1, 3).transpose(-1, -2)
    p = torch.matmul(R.transpose(-1, -2), (q - t.unsqueeze(-1)))
    p = p.transpose(-1, -2).reshape(q_size)
    return p


def apply_rotation_to_vector(R, p):
    """
    Apply a rotation matrix to a vector without translation.

    Parameters
    ----------
    R : torch.Tensor
        Rotation matrix, shape (N, L, 3, 3).
    p : torch.Tensor
        Input vector, shape (N, L, ..., 3).

    Returns
    -------
    torch.Tensor
        Rotated vector, shape (N, L, ..., 3).
    """
    return local2global(R, torch.zeros_like(p), p)


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
def quaternion_to_rotation_matrix(quaternions):
    """
    Convert quaternions to rotation matrices.

    Parameters
    ----------
    quaternions : torch.Tensor
        Quaternions with the real part first, shape (..., 4).

    Returns
    -------
    torch.Tensor
        Rotation matrices, shape (..., 3, 3).
    """
    quaternions = F.normalize(quaternions, dim=-1)
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
def quaternion_1ijk_to_rotation_matrix(q):
    """
    Convert quaternion (1 + ai + bj + ck) to rotation matrix.

    Parameters
    ----------
    q : torch.Tensor
        Quaternion components (b, c, d), shape (..., 3).

    Returns
    -------
    torch.Tensor
        Rotation matrix, shape (..., 3, 3).
    """
    b, c, d = torch.unbind(q, dim=-1)
    s = torch.sqrt(1 + b ** 2 + c ** 2 + d ** 2)
    a, b, c, d = 1 / s, b / s, c / s, d / s

    o = torch.stack(
        (
            a**2 + b**2 - c**2 - d**2, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c,
            2 * b * c + 2 * a * d, a**2 - b**2 + c**2 - d**2, 2 * c * d - 2 * a * b,
            2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a**2 - b**2 - c**2 + d**2,
        ),
        -1,
    )
    return o.reshape(q.shape[:-1] + (3, 3))


def get_consecutive_flag(chain_nb, res_nb, mask):
    """
    Compute flag indicating whether consecutive residues are connected.

    Parameters
    ----------
    chain_nb : torch.Tensor
        Chain indices, shape (N, L).
    res_nb : torch.Tensor
        Residue numbers, shape (N, L).
    mask : torch.Tensor
        Mask indicating valid residues, shape (N, L).

    Returns
    -------
    torch.Tensor
        Boolean tensor indicating connected residues, shape (N, L-1).
    """
    d_res_nb = (res_nb[:, 1:] - res_nb[:, :-1]).abs()
    same_chain = (chain_nb[:, 1:] == chain_nb[:, :-1])
    consec = torch.logical_and(d_res_nb == 1, same_chain)
    consec = torch.logical_and(consec, mask[:, :-1])

    return consec


def get_terminus_flag(chain_nb, res_nb, mask):
    """
    Identify N-terminus and C-terminus flags for residues.

    Parameters
    ----------
    chain_nb : torch.Tensor
        Chain indices, shape (N, L).
    res_nb : torch.Tensor
        Residue numbers, shape (N, L).
    mask : torch.Tensor
        Mask indicating valid residues, shape (N, L).

    Returns
    -------
    tuple of torch.Tensor
        N-terminus and C-terminus flags, both of shape (N, L).
    """
    consec = get_consecutive_flag(chain_nb, res_nb, mask)
    N_term_flag = F.pad(torch.logical_not(consec), pad=(1, 0), value=1)
    C_term_flag = F.pad(torch.logical_not(consec), pad=(0, 1), value=1)
    return N_term_flag, C_term_flag


def dihedral_from_four_points(p0, p1, p2, p3):
    """
    Compute dihedral angle given four points.

    Parameters
    ----------
    p0, p1, p2, p3 : torch.Tensor
        Coordinates of four points, shape (*, 3).

    Returns
    -------
    torch.Tensor
        Dihedral angles in radians, shape (*,).
    """
    v0 = p2 - p1
    v1 = p0 - p1
    v2 = p3 - p2

    u1 = torch.cross(v0, v1, dim=-1)
    n1 = u1 / torch.linalg.norm(u1, dim=-1, keepdim=True)

    u2 = torch.cross(v0, v2, dim=-1)
    n2 = u2 / torch.linalg.norm(u2, dim=-1, keepdim=True)

    sgn = torch.sign((torch.cross(v1, v2, dim=-1) * v0).sum(-1))
    dihed = sgn * torch.acos((n1 * n2).sum(-1).clamp(min=-0.999999, max=0.999999))
    dihed = torch.nan_to_num(dihed)
    return dihed


def get_bb_dihedral_angles(pos_atoms, chain_nb, res_nb, mask_residue):
    """
    Compute backbone dihedral angles (Omega, Phi, Psi) from atomic positions.

    Parameters
    ----------
    pos_atoms : torch.Tensor
        Atomic positions, shape (N, L, A, 3).
    chain_nb : torch.Tensor
        Chain indices, shape (N, L).
    res_nb : torch.Tensor
        Residue numbers, shape (N, L).
    mask_residue : torch.Tensor
        Mask for valid residues, shape (N, L).

    Returns
    -------
    tuple of torch.Tensor
        Backbone dihedral angles and their masks, both of shape (N, L, 3).
    """
    pos_N = pos_atoms[:, :, BBHeavyAtom.N]
    pos_CA = pos_atoms[:, :, BBHeavyAtom.CA]
    pos_C = pos_atoms[:, :, BBHeavyAtom.C]

    N_term_mask, C_term_mask = get_terminus_flag(chain_nb, res_nb, mask_residue)

    omega_mask = torch.logical_not(N_term_mask)
    phi_mask = torch.logical_not(N_term_mask)
    psi_mask = torch.logical_not(C_term_mask)

    omega = F.pad(dihedral_from_four_points(pos_CA[:, :-1], pos_C[:, :-1], pos_N[:, 1:], pos_CA[:, 1:]), pad=(1, 0), value=0)
    phi = F.pad(dihedral_from_four_points(pos_C[:, :-1], pos_N[:, 1:], pos_CA[:, 1:], pos_C[:, 1:]), pad=(1, 0), value=0)
    psi = F.pad(dihedral_from_four_points(pos_N[:, :-1], pos_CA[:, :-1], pos_C[:, :-1], pos_N[:, 1:]), pad=(0, 1), value=0)

    mask_bb_dihed = torch.stack([omega_mask, phi_mask, psi_mask], dim=-1)
    bb_dihed = torch.stack([omega, phi, psi], dim=-1) * mask_bb_dihed

    return bb_dihed, mask_bb_dihed


def pairwise_dihedrals(pos_atoms):
    """
    Compute inter-residue Phi and Psi angles.

    Parameters
    ----------
    pos_atoms : torch.Tensor
        Atomic positions, shape (N, L, A, 3).

    Returns
    -------
    torch.Tensor
        Inter-residue Phi and Psi angles, shape (N, L, L, 2).
    """
    N, L = pos_atoms.size()[:2]
    pos_N = pos_atoms[:, :, BBHeavyAtom.N]
    pos_CA = pos_atoms[:, :, BBHeavyAtom.CA]
    pos_C = pos_atoms[:, :, BBHeavyAtom.C]

    ir_phi = dihedral_from_four_points(
        pos_C[:, :, None, :].expand(N, L, L, 3),
        pos_N[:, None, :, :].expand(N, L, L, 3),
        pos_CA[:, None, :, :].expand(N, L, L, 3),
        pos_C[:, None, :, :].expand(N, L, L, 3)
    )

    ir_psi = dihedral_from_four_points(
        pos_N[:, :, None, :].expand(N, L, L, 3),
        pos_CA[:, :, None, :].expand(N, L, L, 3),
        pos_C[:, :, None, :].expand(N, L, L, 3),
        pos_N[:, None, :, :].expand(N, L, L, 3)
    )

    ir_dihed = torch.stack([ir_phi, ir_psi], dim=-1)

    return ir_dihed


def log_rotation(R):
    """
    Compute the logarithm of a rotation matrix.

    Parameters
    ----------
    R : torch.Tensor
        Rotation matrix, shape (..., 3, 3).

    Returns
    -------
    torch.Tensor
        Logarithm of the rotation matrix, shape (..., 3, 3).
    """
    trace = R[..., range(3), range(3)].sum(-1)
    min_cos = -0.999 if torch.is_grad_enabled() else -1.0
    cos_theta = ((trace - 1) / 2).clamp_min(min=min_cos)
    sin_theta = torch.sqrt(1 - cos_theta ** 2)
    theta = torch.acos(cos_theta)
    coef = ((theta + 1e-8) / (2 * sin_theta + 2e-8))[..., None, None]
    logR = coef * (R - R.transpose(-1, -2))
    return logR


def skewsym_to_so3vec(S):
    """
    Convert a skew-symmetric matrix to an SO(3) vector.

    Parameters
    ----------
    S : torch.Tensor
        Skew-symmetric matrix, shape (..., 3, 3).

    Returns
    -------
    torch.Tensor
        SO(3) vector, shape (..., 3).
    """
    x = S[..., 1, 2]
    y = S[..., 2, 0]
    z = S[..., 0, 1]
    w = torch.stack([x, y, z], dim=-1)
    return w


def exp_skewsym(S):
    """
    Compute the matrix exponential of a skew-symmetric matrix.

    Parameters
    ----------
    S : torch.Tensor
        Skew-symmetric matrix, shape (..., 3, 3).

    Returns
    -------
    torch.Tensor
        Exponential of the skew-symmetric matrix, shape (..., 3, 3).
    """
    x = torch.linalg.norm(skewsym_to_so3vec(S), dim=-1)
    I = torch.eye(3).to(S).view([1 for _ in range(S.dim() - 2)] + [3, 3])
    sinx, cosx = torch.sin(x), torch.cos(x)
    b = (sinx + 1e-8) / (x + 1e-8)
    c = (1 - cosx + 1e-8) / (x ** 2 + 2e-8)
    S2 = S @ S
    return I + b[..., None, None] * S + c[..., None, None] * S2


def so3vec_to_skewsym(w):
    """
    Convert an SO(3) vector to a skew-symmetric matrix.

    Parameters
    ----------
    w : torch.Tensor
        SO(3) vector, shape (..., 3).

    Returns
    -------
    torch.Tensor
        Skew-symmetric matrix, shape (..., 3, 3).
    """
    x, y, z = torch.unbind(w, dim=-1)
    o = torch.zeros_like(x)
    S = torch.stack([o, z, -y, -z, o, x, y, -x, o], dim=-1).reshape(w.shape[:-1] + (3, 3))
    return S


def so3_vec2rotation(w):
    """
    Convert an SO(3) vector to a rotation matrix.

    Parameters
    ----------
    w : torch.Tensor
        SO(3) vector, shape (..., 3).

    Returns
    -------
    torch.Tensor
        Rotation matrix, shape (..., 3, 3).
    """
    return exp_skewsym(so3vec_to_skewsym(w))


def construct_3d_basis(center, p1, p2):
    """
    Construct an orthogonal 3D basis given three points.

    Parameters
    ----------
    center : torch.Tensor
        The center point (C_alpha), shape (N, L, 3).
    p1 : torch.Tensor
        First point (C), shape (N, L, 3).
    p2 : torch.Tensor
        Second point (N), shape (N, L, 3).

    Returns
    -------
    torch.Tensor
        Orthogonal 3D basis matrix, shape (N, L, 3, 3).
    """
    v1 = p1 - center
    e1 = normalize_vector(v1, dim=-1)

    v2 = p2 - center
    u2 = v2 - project_v2v(v2, e1, dim=-1)
    e2 = normalize_vector(u2, dim=-1)

    e3 = torch.cross(e1, e2, dim=-1)

    return torch.cat([e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)], dim=-1)


def rotation_to_so3vec(R):
    """
    Convert a rotation matrix to an SO(3) vector.

    Parameters
    ----------
    R : torch.Tensor
        Rotation matrix, shape (..., 3, 3).

    Returns
    -------
    torch.Tensor
        SO(3) vector, shape (..., 3).
    """
    logR = log_rotation(R)
    return skewsym_to_so3vec(logR)


def random_uniform_so3(size, device='cpu'):
    """
    Generate random SO(3) vectors uniformly from a distribution.

    Parameters
    ----------
    size : list of int
        Size of the output.
    device : str
        Device to create the tensor on (default is 'cpu').

    Returns
    -------
    torch.Tensor
        Random SO(3) vectors, shape (..., 3).
    """
    q = F.normalize(torch.randn(list(size) + [4], device=device), dim=-1)
    R = quaternion_to_rotation_matrix(q)
    return rotation_to_so3vec(R)


def randn_so3(std_idx, angular_distribution, device='cpu'):
    """
    Generate random SO(3) vectors from a normal distribution with specified angular distribution.

    Parameters
    ----------
    std_idx : torch.Tensor
        Indices for the standard deviation, shape (...).
    angular_distribution : ApproxAngularDistribution
        Angular distribution object.
    device : str
        Device to create the tensor on.

    Returns
    -------
    torch.Tensor
        Random SO(3) vectors, shape (..., 3).
    """
    size = std_idx.size()
    u = F.normalize(torch.randn(list(size) + [3], device=device), dim=-1)
    theta = angular_distribution.sample(std_idx)
    return u * theta[..., None]


def reconstruct_backbone(rot, pos, seq, chain_nb, res_nb, mask):
    """
    Reconstruct backbone atoms (N, CA, C, O) from rotations and translations.

    Parameters
    ----------
    rot : torch.Tensor
        Rotation matrices, shape (N, L, 3, 3).
    pos : torch.Tensor
        Positions of backbone atoms, shape (N, L, 3).
    seq : torch.Tensor
        Amino acid sequence indices, shape (N, L).
    chain_nb : torch.Tensor
        Chain numbers, shape (N, L).
    res_nb : torch.Tensor
        Residue numbers, shape (N, L).
    mask : torch.Tensor
        Mask indicating valid residues, shape (N, L).

    Returns
    -------
    torch.Tensor
        Reconstructed backbone atoms, shape (N, L, 4, 3).
    """
    N, L = seq.size()
    bb_coords = backbone_atom_coordinates_tensor.clone().to(pos)
    oxygen_coords = bb_oxygen_coordinate_tensor.clone().to(pos)
    seq = seq.clamp(min=0, max=20)

    bb_coords = bb_coords[seq.flatten()].reshape(N, L, -1, 3)
    oxygen_coords = oxygen_coords[seq.flatten()].reshape(N, L, -1)

    bb_pos = local2global(rot, pos, bb_coords)
    bb_dihedral, _ = get_bb_dihedral_angles(bb_pos, chain_nb, res_nb, mask)

    psi = bb_dihedral[..., 2]
    psi_sin = torch.sin(psi).reshape(N, L, 1, 1)
    psi_cos = torch.cos(psi).reshape(N, L, 1, 1)
    zeros = torch.zeros_like(psi_sin)
    ones = torch.ones_like(psi_sin)

    row1 = torch.cat([ones, zeros, zeros], dim=-1)
    row2 = torch.cat([zeros, psi_cos, -psi_sin], dim=-1)
    row3 = torch.cat([zeros, psi_sin, psi_cos], dim=-1)

    rot_psi = torch.cat([row1, row2, row3], dim=-2)
    rot_psi, pos_psi = compose_rotation_translation([
        (rot, pos),
        (rot_psi, torch.zeros_like(pos)),
    ])

    oxy_pos = local2global(rot_psi, pos_psi, oxygen_coords.reshape(N, L, 1, 3))
    bb_pos = torch.cat([bb_pos, oxy_pos], dim=2)

    return bb_pos


def reconstruct_backbone_partially(pos_atoms, rot_new, pos_new, seq_new, chain_nb, res_nb, mask_atoms, generation_mask):
    """
    Partially reconstruct backbone atoms for generated regions.

    Parameters
    ----------
    pos_atoms : torch.Tensor
        Positions of atoms, shape (N, L, A, 3).
    rot_new : torch.Tensor
        New rotation matrices, shape (N, L, 3, 3).
    pos_new : torch.Tensor
        New positions, shape (N, L, 3).
    seq_new : torch.Tensor
        New amino acid sequence, shape (N, L).
    chain_nb : torch.Tensor
        Chain numbers, shape (N, L).
    res_nb : torch.Tensor
        Residue numbers, shape (N, L).
    mask_atoms : torch.Tensor
        Mask for atoms, shape (N, L, A).
    generation_mask : torch.Tensor
        Mask indicating generated regions, shape (N, L).

    Returns
    -------
    tuple of torch.Tensor
        Updated positions of atoms, shape (N, L, A, 3), and updated mask, shape (N, L, A).
    """
    N, L, A = mask_atoms.size()
    mask_res = mask_atoms[:, :, BBHeavyAtom.CA]

    pos_recons = reconstruct_backbone(rot_new, pos_new, seq_new, chain_nb, res_nb, mask_res)
    pos_recons = F.pad(pos_recons, pad=(0, 0, 0, A - 4), value=0)
    pos_new = torch.where(generation_mask[:, :, None, None].expand_as(pos_atoms), pos_recons, pos_atoms)

    mask_bb_atoms = torch.zeros_like(mask_atoms)
    mask_bb_atoms[:, :, :4] = True
    mask_new = torch.where(generation_mask[:, :, None].expand_as(mask_atoms), mask_bb_atoms, mask_atoms)

    return pos_new, mask_new
