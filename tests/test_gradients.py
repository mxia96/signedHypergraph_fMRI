import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypergraph_construction.config import ConstructionConfig
from hypergraph_construction.construct import construct_hypergraph, infer_hypergraph
from hypergraph_construction.objective import ConstructionObjective, InductiveObjective, row_normalize


def _make_data(seed=0, N=4, V=8, E=4):
    rng = np.random.default_rng(seed)
    H = row_normalize(rng.standard_normal((N, V, E)), 1e-6)
    X = rng.standard_normal((N, 40, V))
    P = np.stack([np.corrcoef(x, rowvar=False) for x in X])
    return H, P


def _finite_diff(f, H, eps=1e-6):
    g = np.zeros_like(H)
    it = np.nditer(H, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        Hp = H.copy(); Hp[idx] += eps
        Hm = H.copy(); Hm[idx] -= eps
        g[idx] = (np.sum(f(Hp)[0]) - np.sum(f(Hm)[0])) / (2 * eps)
        it.iternext()
    return g


def _rel_err(g, g_fd):
    return np.abs(g - g_fd).max() / (np.abs(g_fd).max() + 1e-12)


def test_construction_gradient():
    H, P = _make_data()
    obj = ConstructionObjective(P, alpha=0.7, beta=0.2, lam=0.13)
    rel = _rel_err(obj.value_and_grad(H)[1], _finite_diff(obj.value_and_grad, H))
    print(f"Eq. (7) gradient: max rel err = {rel:.2e}")
    assert rel < 1e-5


def test_inductive_gradient():
    H, P = _make_data()
    S_tr = (H[:3] ** 2).sum(axis=0)
    obj = InductiveObjective(P[3:], S_tr, 3, alpha=0.7, beta=0.2, lam=0.13, eps=1e-6)
    rel = _rel_err(obj.value_and_grad(H[3:])[1], _finite_diff(obj.value_and_grad, H[3:]))
    print(f"inductive objective gradient: max rel err = {rel:.2e}")
    assert rel < 1e-5


def test_inductive_subjects_are_independent():
    _, P = _make_data(N=10)
    cfg = ConstructionConfig(num_hyperedges=4, alpha=0.7, beta=0.2, lam=0.13,
                             max_iters=200, verbose=False)
    H_train, info = construct_hypergraph(P[:6], cfg)
    H_batch, _ = infer_hypergraph(P[6:], H_train, cfg, info["learning_rate"])
    H_single = np.concatenate([infer_hypergraph(P[i:i + 1], H_train, cfg, info["learning_rate"])[0]
                               for i in range(6, 10)])
    assert np.array_equal(H_batch, H_single)
    print("inductive inference: held-out subjects are processed independently")


if __name__ == "__main__":
    test_construction_gradient()
    test_inductive_gradient()
    test_inductive_subjects_are_independent()
    print("All checks passed.")
