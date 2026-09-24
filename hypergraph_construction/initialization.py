"""Common groupwise initialisation H_init (Algorithm 1, steps 1-5)."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from .objective import row_normalize


def kmeans_init(P, num_hyperedges, random_state, eps):
    """k-means on the group-mean connectivity profiles; H_init(i, j) = corr(P_bar(i, :), P_cj)."""
    P_bar = np.asarray(P, dtype=np.float64).mean(axis=0)
    V = P_bar.shape[0]
    labels = KMeans(n_clusters=num_hyperedges, random_state=random_state,
                    n_init=10).fit_predict(P_bar)
    centers = np.zeros((num_hyperedges, P_bar.shape[1]))
    for j in range(num_hyperedges):
        members = labels == j
        if members.any():
            centers[j] = P_bar[members].mean(axis=0)
    H_init = np.nan_to_num(np.corrcoef(P_bar, centers)[:V, V:])
    return row_normalize(H_init, eps)
