"""Objective and gradients for groupwise signed hypergraph construction."""
from __future__ import annotations

import numpy as np

_TINY = 1e-12


def row_normalize(H, eps):
    """Normalise each row to unit l2 norm."""
    norm = np.linalg.norm(H, axis=-1, keepdims=True)
    return H / np.maximum(norm, eps)


def _subject_terms(H, P, B, trPP):
    """Per-subject fidelity, smoothness and diversity terms with their gradients."""
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
    """Groupwise construction"""

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
