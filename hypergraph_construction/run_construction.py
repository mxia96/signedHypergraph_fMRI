"""Build signed hypergraphs for a training set and a test set.

python -m hypergraph_construction.run_construction --config configs/construction.yaml

Writes H_train.npy and H_test.npy (shape (N, |V|, |E|)) to output_dir.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .config import ConstructionConfig
from .construct import construct_hypergraph, infer_hypergraph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = ConstructionConfig.from_yaml(args.config)

    P_train = np.load(cfg.train_correlation).astype(np.float64)
    P_test = np.load(cfg.test_correlation).astype(np.float64)
    print(f"train {P_train.shape}, test {P_test.shape}, setting = {cfg.setting}")

    if cfg.setting == "transductive":
        H_all, info = construct_hypergraph(np.concatenate([P_train, P_test]), cfg, cfg.verbose)
        H_train, H_test = H_all[:len(P_train)], H_all[len(P_train):]
        print(f"construction: {info['iterations']} iterations, J = {info['objective']:.6f}")
    else:
        H_train, info = construct_hypergraph(P_train, cfg, cfg.verbose)
        print(f"construction: {info['iterations']} iterations, J = {info['objective']:.6f}")
        H_test, info_te = infer_hypergraph(P_test, H_train, cfg, info["learning_rate"], cfg.verbose)
        print(f"inference: {info_te['iterations'].mean():.0f} iterations on average")

    os.makedirs(cfg.output_dir, exist_ok=True)
    np.save(os.path.join(cfg.output_dir, "H_train.npy"), H_train.astype(np.float32))
    np.save(os.path.join(cfg.output_dir, "H_test.npy"), H_test.astype(np.float32))
    print(f"saved H_train {H_train.shape} and H_test {H_test.shape} to {cfg.output_dir}")


if __name__ == "__main__":
    main()
