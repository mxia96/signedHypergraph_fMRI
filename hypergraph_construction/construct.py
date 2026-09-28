"""Groupwise signed hypergraph construction."""
from __future__ import annotations

import numpy as np

from .initialization import kmeans_init
from .objective import ConstructionObjective, row_normalize


def learning_rate(cfg, n_subjects, n_rois):
    if cfg.learning_rate is not None:
        return float(cfg.learning_rate)
    return cfg.step_size * n_subjects * n_rois ** 2


def apply_threshold(H, threshold, eps):
    if threshold <= 0:
        return H
    return row_normalize(np.where(np.abs(H) < threshold, 0.0, H), eps)


def _converged(J_new, J_old, eps):
    return np.abs(J_new - J_old) / np.maximum(np.abs(J_old), eps) <= eps


def construct_hypergraph(P, cfg, verbose=False):
    P = np.asarray(P, dtype=np.float64)
    N, V, _ = P.shape
    eps = cfg.tol
    H_init = kmeans_init(P, cfg.num_hyperedges, cfg.init_random_state, eps)
    H = np.repeat(H_init[None], N, axis=0)
    gamma = learning_rate(cfg, N, V)
    objective = ConstructionObjective(P, cfg.alpha, cfg.beta, cfg.lam)

    J, G = objective.value_and_grad(H)
    t = 0
    while t < cfg.max_iters:
        H = row_normalize(H - gamma * G, eps)
        J_new, G = objective.value_and_grad(H)
        t += 1
        done = _converged(J_new, J, eps)
        J = J_new
        if verbose and (t % 100 == 0 or done):
            print(f"[construction] iter {t:5d}  J = {J:.6f}")
        if done:
            break
    return H, {"iterations": t, "objective": float(J), "learning_rate": gamma, "H_init": H_init}
