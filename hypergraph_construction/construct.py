"""Groupwise signed hypergraph construction (Algorithm 1) and inductive inference (Section 2.7)."""
from __future__ import annotations

import numpy as np

from .initialization import kmeans_init
from .objective import ConstructionObjective, InductiveObjective, row_normalize


def learning_rate(cfg, n_subjects, n_rois):
    """Step size gamma of Algorithm 1.

    Every term of Eq. (7) is averaged over subjects and normalised by |V|^2 (or
    |E|^2), so the gradient with respect to one subject's incidence matrix scales
    with 1 / (N |V|^2). Setting gamma = step_size * N * |V|^2 gives every subject
    the same effective step size regardless of cohort size and atlas.
    """
    if cfg.learning_rate is not None:
        return float(cfg.learning_rate)
    return cfg.step_size * n_subjects * n_rois ** 2


def apply_threshold(H, threshold, eps):
    """Set incidence coefficients with |h_ij| < threshold to zero, then re-normalise every row
    to unit l2 norm so that the constraint D~_n = I still holds."""
    if threshold <= 0:
        return H
    return row_normalize(np.where(np.abs(H) < threshold, 0.0, H), eps)


def _converged(J_new, J_old, eps):
    return np.abs(J_new - J_old) / np.maximum(np.abs(J_old), eps) <= eps


def construct_hypergraph(P, cfg, verbose=False):
    """Algorithm 1. P: (N, |V|, |V|) Pearson correlations. Returns H_all (N, |V|, |E|) and a log dict."""
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


def infer_hypergraph(P_test, H_train, cfg, train_learning_rate, verbose=False):
    """Inductive inference for held-out subjects (Section 2.7).

    Each held-out subject starts from the row-normalised mean of the training
    incidence matrices and is optimised with its own connectivity and the fixed
    training quantities only (template, S_tr of Eq. 13). Subjects are updated and
    stopped individually, so the result for one subject does not depend on any
    other held-out subject.
    """
    P_test = np.asarray(P_test, dtype=np.float64)
    H_train = np.asarray(H_train, dtype=np.float64)
    n_train, V, _ = H_train.shape
    eps = cfg.tol
    template = row_normalize(H_train.mean(axis=0), eps)
    S_tr = (H_train ** 2).sum(axis=0)
    # same effective per-subject step as during construction (see learning_rate)
    gamma = train_learning_rate * (n_train + 1) / n_train
    objective = InductiveObjective(P_test, S_tr, n_train, cfg.alpha, cfg.beta, cfg.lam, eps)

    H = np.repeat(template[None], len(P_test), axis=0)
    J, G = objective.value_and_grad(H)
    active = np.ones(len(P_test), dtype=bool)
    iterations = np.zeros(len(P_test), dtype=int)
    t = 0
    while t < cfg.max_iters and active.any():
        H[active] = row_normalize(H[active] - gamma * G[active], eps)
        J_new, G = objective.value_and_grad(H)
        t += 1
        iterations[active] = t
        done = active & _converged(J_new, J, eps)
        J = np.where(active, J_new, J)
        active &= ~done
        if verbose and (t % 100 == 0 or not active.any()):
            print(f"[inference] iter {t:5d}  active subjects = {int(active.sum())}")
    return H, {"iterations": iterations, "objective": J, "learning_rate": gamma}
