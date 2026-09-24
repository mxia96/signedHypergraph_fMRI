"""Construction objective (Eq. 7) and the training-anchored inductive objective (Eqs. 13-14).

All terms are evaluated for a stack of subjects H with shape (N, |V|, |E|) and
Pearson correlation matrices P with shape (N, |V|, |V|). Gradients are analytic.
"""
from __future__ import annotations

import numpy as np

_TINY = 1e-12


def row_normalize(H, eps):
    """D~_n = I: scale every row (node) of every incidence matrix to unit l2 norm."""
    norm = np.linalg.norm(H, axis=-1, keepdims=True)
    return H / np.maximum(norm, eps)


def _subject_terms(H, P, B, trPP):
    """Per-subject fidelity, Laplacian smoothness and hyperedge diversity, with gradients.

    Returns un-normalised values (length-N arrays) and gradients (N, |V|, |E|) of
      fidelity   ||P_n - H_n H_n^T||_F^2
      smoothness tr(P_n^T L_n P_n),  L_n = I - D_n^{-1/2} H_n Delta_n^{-1} H_n^T D_n^{-1/2}
      diversity  ||Delta~_n^{-1} H_n^T H_n Delta~_n^{-1} - I||_F^2
    D_n and Delta_n use absolute incidence weights; Delta~_n uses column l2 norms.
    """
    E = H.shape[2]
    Ht = np.transpose(H, (0, 2, 1))

    R = H @ Ht - P
    fid = np.einsum("nij,nij->n", R, R)
    g_fid = 4.0 * (R @ H)

    absH, sgn = np.abs(H), np.sign(H)
    dv = np.maximum(absH.sum(axis=2), _TINY)
    de = np.maximum(absH.sum(axis=1), _TINY)
    a = dv ** -0.5
    C = a[:, :, None] * B * a[:, None, :]
    HD = H / de[:, None, :]
    lap = trPP - np.einsum("nij,nij->n", C, HD @ Ht)
    CH = C @ H
    g_h = 2.0 * CH / de[:, None, :]
    g_de = -(np.einsum("nve,nve->ne", H, CH) / de ** 2)[:, None, :] * sgn
    phi = np.einsum("nve,nve->nv", HD, B @ (a[:, :, None] * H))
    g_dv = -(dv ** -1.5 * phi)[:, :, None] * sgn
    g_lap = -(g_h + g_dv + g_de)

    r = np.maximum(np.linalg.norm(Ht, axis=2, keepdims=True), _TINY)
    U = Ht / r
    Sim = U @ np.transpose(U, (0, 2, 1))
    S = Sim.copy()
    S[:, np.arange(E), np.arange(E)] = 0.0
    div = np.einsum("nij,nij->n", S, S)
    w = np.einsum("nab,nab->na", S, Sim)[:, :, None]
    g_div = np.transpose((4.0 / r) * (S @ U - w * U), (0, 2, 1))

    return fid, g_fid, lap, g_lap, div, g_div


class ConstructionObjective:
    """J(H_all) of Eq. (7) for the N subjects used in construction."""

    def __init__(self, P, alpha, beta, lam):
        self.P = np.asarray(P, dtype=np.float64)
        self.B = self.P @ np.transpose(self.P, (0, 2, 1))
        self.trPP = np.einsum("nij,nij->n", self.P, self.P)
        self.alpha, self.beta, self.lam = alpha, beta, lam

    def value_and_grad(self, H):
        N, V, E = H.shape
        fid, g_fid, lap, g_lap, div, g_div = _subject_terms(H, self.P, self.B, self.trPP)
        c_fid = 1.0 / (N * V ** 2)
        c_lap = self.alpha / (N * V ** 2)
        c_div = self.beta / (N * E ** 2)
        c_l21 = self.lam / (np.sqrt(N) * V * np.sqrt(E))
        s = np.sqrt((H ** 2).sum(axis=0))
        J = c_fid * fid.sum() + c_lap * lap.sum() + c_div * div.sum() + c_l21 * s.sum()
        G = (c_fid * g_fid + c_lap * g_lap + c_div * g_div
             + c_l21 * H / np.maximum(s, _TINY)[None])
        return J, G


class InductiveObjective:
    """Per-subject objective for held-out subjects (Section 2.7).

    For each held-out subject s, the fidelity, smoothness and diversity terms of
    Eq. (7) are kept and the groupwise sparsity term is replaced by the
    training-anchored term R_ind of Eq. (14), with S_tr of Eq. (13) fixed. The
    normalisation constants are those of Eq. (7) for a cohort consisting of the
    N_tr training subjects plus subject s, so that every term has the same weight
    per subject as during construction. Held-out subjects are never coupled.
    """

    def __init__(self, P, S_tr, n_train, alpha, beta, lam, eps):
        self.P = np.asarray(P, dtype=np.float64)
        self.B = self.P @ np.transpose(self.P, (0, 2, 1))
        self.trPP = np.einsum("nij,nij->n", self.P, self.P)
        self.S_tr = np.asarray(S_tr, dtype=np.float64)
        self.n = n_train + 1
        self.alpha, self.beta, self.lam, self.eps = alpha, beta, lam, eps

    def value_and_grad(self, H):
        """Returns the objective of every held-out subject (length N_test) and its gradient."""
        _, V, E = H.shape
        fid, g_fid, lap, g_lap, div, g_div = _subject_terms(H, self.P, self.B, self.trPP)
        c_fid = 1.0 / (self.n * V ** 2)
        c_lap = self.alpha / (self.n * V ** 2)
        c_div = self.beta / (self.n * E ** 2)
        c_anc = self.lam / (np.sqrt(self.n) * V * np.sqrt(E))
        root = np.sqrt(self.S_tr[None] + H ** 2 + self.eps)
        r_ind = (root - np.sqrt(self.S_tr + self.eps)[None]).sum(axis=(1, 2))
        J = c_fid * fid + c_lap * lap + c_div * div + c_anc * r_ind
        G = c_fid * g_fid + c_lap * g_lap + c_div * g_div + c_anc * H / root
        return J, G
